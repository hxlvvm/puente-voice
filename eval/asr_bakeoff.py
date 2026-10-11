"""ASR bake-off spike: Whisper turbo vs Parakeet v3 on 8 kHz phone audio (real MSWC words, FLEURS sentences, Kokoro clips,
noise). Prints accuracy, WER, empty-on-noise and latency per engine. Synthetic + CC-BY public data only."""
import asyncio
import csv
import glob
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import soxr
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm

from puente_voice.sim.audio_sim import wer_counts
from puente_voice.text import fold
from puente_voice.voice.speech import load_kokoro, synth

V = sys.argv[1]
E = f"{V}/eval"
rng = np.random.default_rng(0)


async def _phone(pcm16, rate):
    """int16 PCM at `rate` -> 8 kHz mu-law -> back -> float32 16 kHz segment with 1 s lead / 0.5 s tail (pipeline shape)."""
    if rate != 8000:
        pcm16 = soxr.resample(np.frombuffer(pcm16, np.int16), rate, 8000).astype(np.int16).tobytes()
    u = await pcm_to_ulaw(pcm16, 8000, 8000, create_stream_resampler())
    back = await ulaw_to_pcm(u, 8000, 8000, create_stream_resampler())
    x = np.frombuffer(back, np.int16).astype(np.float32) / 32768.0
    x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(4000, np.float32)])
    return soxr.resample(x, 8000, 16000).astype(np.float32)


def phone(pcm16, rate):
    return asyncio.run(_phone(pcm16, rate))


def wav_pcm(path):
    x, sr = sf.read(path, dtype="float32")  # FLEURS wavs are float: an int16 read zeroes them
    if x.ndim > 1:
        x = x[:, 0]
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(), sr


def norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", fold(t)))


# --- test sets -------------------------------------------------------------------------------------
words = []  # (set, clip, reference)
for p in sorted(glob.glob(f"{E}/words/*.wav")):
    w = os.path.basename(p).split("_common_voice")[0]
    if w in {"claro", "vale", "bueno", "hola", "gracias", "correcto", "primera", "segunda"} and sum(1 for s, _, r in words if r == w) < 25:  # cap per word
        words.append(("real-word", phone(*wav_pcm(p)), w))
sentences = []
rows = list(csv.reader(open(f"{E}/fleurs/test.tsv", encoding="utf-8"), delimiter="\t"))
by_name = {r[1]: r[2] for r in rows}
for p in sorted(glob.glob(f"{E}/sentences/test/*.wav"))[:80]:
    sentences.append(("real-sentence", phone(*wav_pcm(p)), by_name[os.path.basename(p)]))
k = load_kokoro(f"{V}/kokoro", int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
synthetic = []
for voice in ("em_alex", "em_santa", "ef_dora"):
    for text in ("sí", "no", "claro", "hola", "la primera", "sí, correcto"):
        pcm, sr = synth(k, text, "es", voice)
        synthetic.append(("kokoro", phone(pcm, sr), text))
level = 2500.0
noise = [("noise", phone(np.clip(rng.normal(0, level / 10 ** 0.75, 16000), -32768, 32767).astype(np.int16).tobytes(), 8000), ""),
         ("noise", phone((np.hanning(2000) * rng.normal(0, 6000, 2000)).astype(np.int16).tobytes(), 8000), "")]
print(f"sets: {len(words)} real words, {len(sentences)} sentences, {len(synthetic)} kokoro, {len(noise)} noise", flush=True)


# --- engines ---------------------------------------------------------------------------------------
def whisper_engine(device, threads, size="deepdml/faster-whisper-large-v3-turbo-ct2"):
    from faster_whisper import WhisperModel
    m = WhisperModel(size, device=device,
                     compute_type="float16" if device == "cuda" else "int8", cpu_threads=threads)

    def run(x):
        segs, _ = m.transcribe(x, language="es", vad_filter=True)
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.4)
    return run


def parakeet_engine(threads):
    import sherpa_onnx
    d = glob.glob(f"{V}/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")[0]
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
        tokens=f"{d}/tokens.txt", num_threads=threads, model_type="nemo_transducer", decoding_method="greedy_search")

    def run(x):
        s = rec.create_stream()
        s.accept_waveform(16000, x)
        rec.decode_stream(s)
        return s.result.text.strip()
    return run


def evaluate(name, run, subset):
    stats = {}
    for kind, x, ref in subset:
        t = time.perf_counter()
        hyp = run(x)
        dt = time.perf_counter() - t
        st = stats.setdefault(kind, {"n": 0, "ok": 0, "edits": 0, "words": 0, "lat": [], "bad": []})
        st["n"] += 1
        st["lat"].append(dt)
        if kind == "real-sentence":
            e, w = wer_counts(ref, hyp)
            st["edits"] += e
            st["words"] += w
        elif kind == "noise":
            st["ok"] += norm(hyp) == ""
            if norm(hyp):
                st["bad"].append(hyp)
        else:
            good = norm(hyp) == norm(ref)
            st["ok"] += good
            if not good and len(st["bad"]) < 6:
                st["bad"].append(f"{ref}->{hyp!r}")
    for kind, st in stats.items():
        lat = np.array(st["lat"])
        score = (f"WER {st['edits'] / st['words']:.1%}" if kind == "real-sentence" else f"acc {st['ok']}/{st['n']}")
        print(f"{name:14} {kind:14} {score:14} lat mean {lat.mean():.3f}s p95 {np.percentile(lat, 95):.3f}s  "
              f"{'; '.join(st['bad'][:6])}", flush=True)


full = words + sentences + synthetic + noise
small = words[::5] + sentences[:20] + synthetic + noise  # CPU Whisper is slow: a subset
evaluate("whisper-gpu", whisper_engine("cuda", 4), full)
evaluate("parakeet-cpu4", parakeet_engine(4), full)
evaluate("parakeet-cpu2", parakeet_engine(2), small)
evaluate("whisper-small-cpu4", whisper_engine("cpu", 4, "small"), small)
evaluate("whisper-medium-cpu4", whisper_engine("cpu", 4, "medium"), small)
