"""End-to-end smoke test through a real Asterisk 22 (Apptainer image of the compose service), no SIP phone needed.

Asterisk originates Local calls into the demo dialplan (deploy/asterisk/extensions.conf); the far end of each call
plays Kokoro-spoken caller lines and records what it hears. Three calls: asks for staff, reports chest pain, hangs
up after choosing the language. Checks the dialplan outcome, the audit log, agent audio and response latency.

usage: python eval/asterisk_smoke.py ASTERISK_SIF MODELS_DIR WORKDIR [cuda|cpu]   (jobs/asterisk_smoke.sbatch)
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NOTICE_S = 37  # the opening notice runs 35 s
LINES = {"espanol": "Español, por favor.", "persona": "Quiero hablar con una persona.",
         "pecho": "Me duele mucho el pecho y no puedo respirar bien."}
CALLS = {  # name: caller lines in order; None = hang up
    "transfer": ["espanol", "persona"], "emergency": ["espanol", "pecho"], "hangup": ["espanol", None]}


def wavs(models, out):
    """One continuous recording per call, as a phone line streams audio even in silence (Wait() would send none)."""
    from pipecat.audio.utils import create_stream_resampler

    from puente_voice.voice.speech import load_kokoro, synth
    k, spoken = load_kokoro(f"{models}/kokoro", 4), {}
    for name, text in LINES.items():
        pcm, sr = synth(k, text, "es", "em_alex")
        spoken[name] = np.frombuffer(asyncio.run(create_stream_resampler().resample(pcm, sr, 8000)), np.int16)
    rng, ends = np.random.default_rng(0), {}
    for name, lines in CALLS.items():
        parts, t = [np.zeros(NOTICE_S * 8000, np.int16)], NOTICE_S
        for line in lines:
            if line is None:
                break
            parts += [spoken[line], np.zeros(8 * 8000, np.int16)]
            ends[name] = t + len(spoken[line]) / 8000  # when the caller's last line ends
            t += len(spoken[line]) / 8000 + 8
        if lines[-1]:
            parts.append(np.zeros(20 * 8000, np.int16))  # time for the agent's answer
        x = np.concatenate(parts)
        x = (x + rng.normal(0, 30, len(x))).clip(-32768, 32767).astype(np.int16)  # faint line noise
        with wave.open(str(out / f"call-{name}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000), w.writeframes(x.tobytes())
        ends.setdefault(name, None)
    return ends


def pjsip(out, port):
    """deploy/asterisk/pjsip.conf.in rendered by the same sed line compose runs, bound to a free port (shared node)."""
    text = subprocess.run(["sed", "-e", "s/@SIP_PASSWORD@/smoketest123/", "-e", "s/@PUBLIC_IP@/127.0.0.1/",
                           str(REPO / "deploy/asterisk/pjsip.conf.in")], check=True, capture_output=True, text=True).stdout
    assert "@" not in text.replace("asterisk", "")
    (out / "pjsip.conf").write_text(text.replace("bind=0.0.0.0:5060", f"bind=127.0.0.1:{port}"))


def dialplan(out):
    """The demo dialplan plus a caller-sim context that plays each call's recording and records the call."""
    text = (REPO / "deploy/asterisk/extensions.conf").read_text() + "\n[caller-sim]\n"
    for name in CALLS:
        text += (f"exten => {name},1,Answer()\n same => n,MixMonitor({out}/heard-{name}.wav)\n"
                 f" same => n,Playback({out}/call-{name})\n same => n,Hangup()\n")
    (out / "extensions.conf").write_text(text)
    (out / "logger.conf").write_text("[general]\n[logfiles]\nfull => notice,warning,error,verbose(3)\n")


def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def rms_track(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    return np.sqrt(np.mean(x[: len(x) // 160 * 160].reshape(-1, 160) ** 2, axis=1))  # per 20 ms


def main(sif, models, work, device="cuda"):
    out = Path(work).resolve()
    shutil.rmtree(out, ignore_errors=True)
    for d in ("spool", "log", "tmp"):
        (out / d).mkdir(parents=True)
    ends = wavs(models, out)
    dialplan(out)
    base = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 10000 * 2  # shared node: our own ports
    http, audio = base, base + 1
    pjsip(out, base + 2)
    env = {**os.environ, "PUENTE_MODELS": models, "PUENTE_DEVICE": device, "PUENTE_AUDIT": str(out / "audit.jsonl"),
           "PUENTE_HTTP_PORT": str(http), "PUENTE_AUDIOSOCKET_PORT": str(audio)}
    agent = subprocess.Popen([sys.executable, "-m", "puente_voice.serve"], env=env, cwd=REPO,
                             stdout=open(out / "agent.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while not (port_open(http) and port_open(audio)):
        assert agent.poll() is None, "agent exited: see agent.log"
        time.sleep(1)
    print(f"agent ready {time.time() - t0:.0f} s after start ({device})", flush=True)
    binds = [f"{out}/{f}:/etc/asterisk/{f}" for f in ("extensions.conf", "pjsip.conf", "logger.conf")]
    binds.append(f"{REPO}/deploy/asterisk/rtp.conf:/etc/asterisk/rtp.conf")
    binds += [f"{out}/spool:/var/spool/asterisk/outgoing", f"{out}/log:/var/log/asterisk", f"{out}:{out}"]
    pbx = subprocess.Popen(["apptainer", "exec", "--writable-tmpfs", "--cleanenv",
                            "--env", f"PUENTE_AGENT=127.0.0.1:{audio},PUENTE_AGENT_HTTP=127.0.0.1:{http}",
                            *sum((["--bind", b] for b in binds), []), sif, "asterisk", "-f", "-vvv"],
                           stdout=open(out / "asterisk.out", "w"), stderr=subprocess.STDOUT)
    time.sleep(8)
    assert pbx.poll() is None, "asterisk exited: see asterisk.out"
    try:
        for name, lines in CALLS.items():
            call = out / "tmp" / f"{name}.call"
            call.write_text(f"Channel: Local/100@puente-in\nContext: caller-sim\nExtension: {name}\nPriority: 1\n")
            os.replace(call, out / "spool" / call.name)  # atomic, as the spooler requires
            time.sleep(NOTICE_S + 8 * len(lines) + 30)
    finally:
        pbx.terminate(), agent.terminate()
        pbx.wait(20), agent.wait(20)

    full = (out / "log/full").read_text(errors="replace")
    outcomes = [line.split("PUENTE_OUTCOME", 1)[1].strip() for line in full.splitlines()
                if "PUENTE_OUTCOME" in line and "Executing" not in line]  # the Verbose() output, not its trace
    events = [json.loads(line) for line in open(out / "audit.jsonl")] if (out / "audit.jsonl").exists() else []
    calls = {}
    for e in events:
        calls.setdefault(e["call_id"], []).append({k: v for k, v in e.items() if k not in ("ts", "call_id")})
    print("dialplan outcomes, in call order:", outcomes)
    for i, (cid, ev) in enumerate(calls.items()):
        print(f"call {i + 1} audit:", ev)
    expected = ("Could not create dialog", "Failed to create outgoing session")  # no staff phone in the test to ring
    sip_errors = [line for line in full.splitlines()
                  if "ERROR" in line and "pjsip" in line.lower() and not any(e in line for e in expected)]
    print("pjsip errors:", sip_errors or "none")
    ok = outcomes == ["transferred", "transferred"] and len(calls) == 3 and not sip_errors
    for name, lines in CALLS.items():
        r = rms_track(out / f"heard-{name}.wav")
        notice = (r[: 30 * 50] > 300).mean()
        print(f"{name}: agent audio in the first 30 s on {notice:.0%} of 20 ms frames")
        ok &= notice > 0.3
        if lines[-1]:  # response latency: end of the caller's last line to the agent's next voiced frame
            after = np.nonzero(r[int(ends[name] * 50) + 5:] > 300)[0]
            print(f"{name}: agent answered {after[0] / 50 + 0.1:.2f} s after the caller stopped" if len(after) else
                  f"{name}: agent never answered")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
