"""Simulated phone calls: a scripted caller speaks over the Twilio Media Streams protocol to the real pipeline.

  python -m puente_voice.sim.audio_sim --models DIR scenarios scenarios/audio

The caller is a second Kokoro voice at 8 kHz μ-law, so WER here is a lower bound for real callers, not a field
number. Exit code 1 when a gate fails: a scenario failed, a wrong booking, or p95 latency over 2.5 s.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed

from puente_voice.text import fold

CALLER_VOICE = {"es": "ef_dora", "en": "am_michael"}  # es: eval/voice_match.py (job 7619172); no voice distinguishable, ef_dora by one take
FRAME = 160  # 20 ms of 8 kHz μ-law
VOICED_RMS = 200  # agent audio louder than this counts as speech
QUIET_S = 1.0  # the agent has finished once it is silent this long
PAUSE_S = 0.4
P50_BUDGET_S, P95_GATE_S = 1.5, 2.5  # spec latency budget; only p95 is a gate


def _words(text):
    return re.findall(r"[a-z0-9']+", fold(text))


def wer_counts(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1], len(r)


def rms(pcm):
    x = np.frombuffer(pcm, np.int16).astype(np.float64)
    return float(np.sqrt((x ** 2).mean())) if x.size else 0.0


def add_noise(pcm, noise_rms, rng):
    x = np.frombuffer(pcm, np.int16).astype(np.float64) + rng.normal(0, noise_rms, len(pcm) // 2)
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


def last_speech_span(voiced, gap=QUIET_S):
    """Seconds of the last stretch of agent speech: voiced frames no more than `gap` apart."""
    if not voiced:
        return 0.0
    i = len(voiced) - 1
    while i > 0 and voiced[i] - voiced[i - 1] <= gap:
        i -= 1
    return voiced[-1] - voiced[i] + 0.04  # + one 40 ms output chunk


def wer_reference(caller, reached):
    """What the caller actually said: the first `reached` scripted turns, silent ones skipped, pauses as spaces."""
    return " ".join(u.replace("|", " ") for u in caller[:reached] if u)


def utterance_pcm(kokoro, text, lang):
    """8 kHz 16-bit PCM of the caller saying `text`; each `|` is a 0.4 s pause."""
    import soxr

    from puente_voice.voice.speech import synth
    pause = b"\x00\x00" * int(8000 * PAUSE_S)
    parts = []
    for part in text.split("|"):
        pcm, rate = synth(kokoro, part.strip(), lang, CALLER_VOICE[lang])
        x = soxr.resample(np.frombuffer(pcm, np.int16), rate, 8000)  # band-limited, as a phone line is
        parts.append(np.asarray(x, dtype=np.int16).tobytes())
    return pause.join(parts)


async def _ulaw_frames(pcm):
    from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw
    u = await pcm_to_ulaw(pcm, 8000, 8000, create_stream_resampler())
    u += b"\xff" * (-len(u) % FRAME)
    return [u[i:i + FRAME] for i in range(0, len(u), FRAME)]


class Caller:
    """One phone line: paces 20 ms frames in real time and timestamps the agent's voiced audio."""

    def __init__(self, ws, idle_frames):
        self.ws, self.idle, self.voiced, self.closed, self._k = ws, idle_frames, [], False, 0

    async def start(self, sid):
        await self.ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await self.ws.send(json.dumps({"event": "start", "streamSid": f"MZ{sid}", "start": {
            "streamSid": f"MZ{sid}", "callSid": sid, "accountSid": "ACsim", "tracks": ["inbound"],
            "customParameters": {}, "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))
        self.sid = f"MZ{sid}"

    async def listen(self):
        from pipecat.audio.utils import create_stream_resampler, ulaw_to_pcm
        dec = create_stream_resampler()
        try:
            async for message in self.ws:
                m = json.loads(message)
                if m.get("event") == "media":
                    pcm = await ulaw_to_pcm(base64.b64decode(m["media"]["payload"]), 8000, 8000, dec)
                    if rms(pcm) > VOICED_RMS:
                        self.voiced.append(time.perf_counter())
        except Exception:  # the server closing the socket ends the call
            pass
        finally:
            self.closed = True

    async def send(self, frames):
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            if self.closed:
                return
            try:
                await self.ws.send(json.dumps({"event": "media", "streamSid": self.sid,
                                               "media": {"payload": base64.b64encode(f).decode()}}))
            except ConnectionClosed:  # the agent hung up mid-frame: the call is over
                self.closed = True
                return
            await asyncio.sleep(max(0.0, t0 + 0.02 * (i + 1) - time.perf_counter()))

    async def wait(self, done, timeout):
        """Keep the line open (silence or noise) until done() or timeout; the VAD needs continuous audio."""
        end = time.perf_counter() + timeout
        while not self.closed and not done() and time.perf_counter() < end:
            block = [self.idle[(self._k + i) % len(self.idle)] for i in range(10)]
            self._k += 10
            await self.send(block)

    def first_voice(self, after):
        return next((t for t in self.voiced if t > after), None)

    def quiet_since(self, after):
        last = self.voiced[-1] if self.voiced else 0.0
        return last > after and time.perf_counter() - last > QUIET_S


async def run_call(port, spec, kokoro, calls, sched):
    import websockets
    from puente_voice.sim.text_sim import check
    from puente_voice.voice.speech import synth
    lang, sid = spec.get("lang", "es"), f"CA-{spec['name']}"
    pcms = [await asyncio.to_thread(utterance_pcm, kokoro, u, lang) if u else b"" for u in spec["caller"]]
    noise = 0.0
    if "noise_snr_db" in spec:
        noise = float(np.mean([rms(p) for p in pcms if p])) / 10 ** (spec["noise_snr_db"] / 20)
    rng = np.random.default_rng(0)
    idle = await _ulaw_frames(add_noise(b"\x00\x00" * 8000 * 10, noise, rng) if noise else b"\x00\x00" * 8000)
    turns = [await _ulaw_frames(add_noise(p, noise, rng) if noise else p) if p else [] for p in pcms]
    cough = await _ulaw_frames((np.hanning(2000) * rng.normal(0, 6000, 2000)).astype(np.int16).tobytes())  # 0.25 s
    latencies, barge, take = [], spec.get("barge_in"), spec.get("take_slot")
    async with websockets.connect(f"ws://127.0.0.1:{port}/twilio") as ws:
        c = Caller(ws, idle)
        await c.start(sid)
        rx = asyncio.create_task(c.listen())
        await c.wait(lambda: c.quiet_since(0.0), timeout=90)  # the disclosure
        t_end = time.perf_counter()
        reached = 0
        for i, frames in enumerate(turns):
            if c.closed:  # the agent ended the call; later turns are never spoken
                break
            reached = i + 1
            if take and take["before_turn"] == i:
                sched.take(take["slot"])
            t = time.perf_counter()
            if not frames:  # a silent turn: wait for the idle re-prompt to finish
                await c.wait(lambda: c.quiet_since(t), timeout=40)
                continue
            await c.send(frames)
            t_end = time.perf_counter()
            await c.wait(lambda: c.first_voice(t_end) is not None, timeout=15)
            first = c.first_voice(t_end)
            if first and spec.get("cough_in_reply") == i:  # a cough 1 s into the agent's reply to this turn
                await c.wait(lambda: time.perf_counter() - first > 1.0, timeout=5)
                await c.send(cough)
            if first and i != barge:
                latencies.append(first - t_end)
            if i + 1 == barge:  # the next turn talks over this reply
                await c.wait(lambda: time.perf_counter() - (first or t_end) > 1.0, timeout=5)
            else:
                await c.wait(lambda: c.quiet_since(t_end), timeout=60)
        await c.wait(lambda: False, timeout=2)
        heard_s = last_speech_span(c.voiced) if c.first_voice(t_end) else 0.0  # the final prompt, from its own start
        if not c.closed:
            await ws.send(json.dumps({"event": "stop", "streamSid": c.sid}))
        rx.cancel()
    fp = calls[sid]
    problems = check(spec, fp.flow, " ".join(fp.said))
    if fp.flow.state.value in ("done", "transferred"):  # the goodbye or 911 script must play in full
        pcm, rate = await asyncio.to_thread(synth, kokoro, fp.said[-1], fp.flow.lang)
        if heard_s < 0.8 * len(pcm) / 2 / rate:
            problems.append(f"final prompt cut short: heard {heard_s:.1f}s of {len(pcm) / 2 / rate:.1f}s")
    edits, words = wer_counts(wer_reference(spec["caller"], reached), " ".join(fp.heard))
    booked = fp.flow.booking.slot_id if fp.flow.booking else None
    return {"name": spec["name"], "lang": lang, "passed": not problems, "problems": problems,
            "latencies": latencies, "wer_edits": edits, "wer_words": words, "booked_slot": booked,
            "wrong_booking": booked is not None and booked != spec.get("expect", {}).get("booked_slot"),
            "state": fp.flow.state.value, "transfer_reason": fp.flow.transfer_reason,
            "said": fp.said, "heard": fp.heard, "caller": spec["caller"]}


async def run_all(specs, models, whisper, device, idle_s, log_path, port=8765, safety_model_dir=None):
    import uvicorn
    from faster_whisper import WhisperModel

    from puente_voice.audit import AuditLog
    from puente_voice.flow import CallFlow
    from puente_voice.render import Renderer
    from puente_voice.sim import demo_data
    from puente_voice.voice.server import create_app
    from puente_voice.voice.speech import load_kokoro, prerender

    kokoro = load_kokoro(f"{models}/kokoro", int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    pre, rate = prerender(kokoro, Renderer(demo_data.CLINIC))
    model = WhisperModel(whisper, device=device, compute_type="float16" if device == "cuda" else "int8")
    safety_model = None
    if safety_model_dir:
        from puente_voice.safety_model import SafetyModel
        safety_model = SafetyModel(safety_model_dir)
    current, calls = {}, {}
    app = create_app(lambda cid: CallFlow(cid, demo_data.CLINIC, current["sched"], AuditLog(),
                                          now=lambda: demo_data.NOW, safety_model=safety_model),
                     model, kokoro, pre, rate, idle_s=idle_s, calls=calls)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.1)
    results = []
    try:
        for spec in specs:
            current["sched"] = demo_data.scheduler()
            results.append(await run_call(port, spec, kokoro, calls, current["sched"]))
            r = results[-1]
            with open(log_path, "a", encoding="utf-8") as f:  # kept even if a later call crashes
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']:<24} "
                  f"{r['booked_slot'] or r['transfer_reason'] or r['state']}  {'; '.join(r['problems'])}", flush=True)
    finally:
        server.should_exit = True
        await serving
    return results


def report(results):
    lat = [x for r in results for x in r["latencies"]]
    p50, p95 = (float(np.percentile(lat, q)) for q in (50, 95)) if lat else (float("nan"),) * 2
    wer = {}
    for lang in ("es", "en"):
        rs = [r for r in results if r["lang"] == lang]
        words = sum(r["wer_words"] for r in rs)
        wer[lang] = (sum(r["wer_edits"] for r in rs) / words if words else None, words)
    failed, wrong = sum(not r["passed"] for r in results), sum(r["wrong_booking"] for r in results)
    lines = [f"{len(results) - failed}/{len(results)} scenarios passed; wrong bookings {wrong}",
             f"latency p50 {p50:.2f}s (budget {P50_BUDGET_S}s: {'met' if p50 <= P50_BUDGET_S else 'MISSED'}), "
             f"p95 {p95:.2f}s (gate {P95_GATE_S}s) over {len(lat)} turns"]
    lines += [f"WER {lang} {w:.1%} over {n} words (synthetic TTS speech, 8 kHz mu-law)"
              for lang, (w, n) in wer.items() if w is not None]
    ok = failed == 0 and wrong == 0 and lat and p95 <= P95_GATE_S
    return ok, lines, {"p50": p50, "p95": p95, "wer": wer, "failed": failed, "wrong_bookings": wrong}


def main(argv=None):
    from puente_voice.voice.server import configure_logging
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="scenario directories, or calls-*.jsonl files with --merge")
    ap.add_argument("--models")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th scenario starting at i (array jobs)")
    ap.add_argument("--merge", action="store_true", help="report over the per-call results of several shards")
    ap.add_argument("--whisper", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--idle", type=float, default=6.0)
    ap.add_argument("--out", default="runs/audio")
    ap.add_argument("--log-level", default="INFO", help="DEBUG prints transcripts: synthetic scenarios only")
    ap.add_argument("--safety-model", help="safety classifier dir, ORed with the rules (CPU)")
    a = ap.parse_args(argv)
    if a.merge:
        ok, lines, _ = report([json.loads(line) for f in a.dirs for line in open(f, encoding="utf-8") if line.strip()])
        print("\n".join(lines))
        return 0 if ok else 1
    if not a.models:
        ap.error("--models is required to run calls")
    from puente_voice.sim.text_sim import load_scenarios
    configure_logging(a.log_level)
    i, n = map(int, a.shard.split("/"))
    specs = [s for d in a.dirs for s in load_scenarios(d)][i::n]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    run = os.environ.get("SLURM_ARRAY_JOB_ID") or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"  # one name per array
    stamp = f"{run}-s{i}of{n}"
    results = asyncio.run(run_all(specs, a.models, a.whisper, a.device, a.idle, Path(a.out) / f"calls-{stamp}.jsonl",
                                  port=8765 + i, safety_model_dir=a.safety_model))  # shards sharing a node need their own port
    ok, lines, summary = report(results)
    print("\n".join(lines))
    path = Path(a.out) / f"results-{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print(f"results: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
