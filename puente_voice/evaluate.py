"""Meaning-level evaluation: does the call flow get what the caller meant? Used by the eval/ scripts."""
import math

import yaml


def meaning(ip, kind, text, lang):
    if kind == "name":
        return ip.name(text)
    if kind == "dob":
        d = ip.dob(text, lang)
        return d.isoformat() if d else None
    if kind == "visit":
        return ip.visit(text)
    if kind == "choice":
        return ip.choice(text, 3)
    if kind == "yes_no":
        return ip.yes_no(text)
    if kind == "language":
        return ip.language(text)
    if kind == "persona":
        return ip.persona(text)
    raise ValueError(f"unknown kind {kind!r}")


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_set(path, split=None):
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f)
    return [i for i in items if split in (None, i["split"])]


def score(items, ip, heard=None):
    out = {}
    for it in items:
        text = heard(it) if heard else it["text"]
        got = meaning(ip, it["kind"], text, it["lang"])
        row = out.setdefault(it["kind"], [0, 0, []])
        row[1] += 1
        if got == it["expected"]:
            row[0] += 1
        else:
            row[2].append(f"{it['text']!r} heard {text!r} -> {got!r}, want {it['expected']!r}")
    return out


def report(scores):
    lines = []
    for kind, (ok, n, misses) in sorted(scores.items()):
        lo, hi = wilson(ok, n)
        lines.append(f"{kind:9} {ok:3}/{n:<3} {ok / n:6.1%}  (95 % CI {lo:.0%}-{hi:.0%})")
        lines += [f"           miss: {m}" for m in misses]
    return lines


def phone_segment(pcm16, rate):
    """int16 PCM -> 8 kHz mu-law and back -> float32 at 16 kHz with 1 s lead / 0.5 s tail, as the live pipeline feeds STT."""
    import asyncio

    import numpy as np
    import soxr
    from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm

    if rate != 8000:
        pcm16 = soxr.resample(np.frombuffer(pcm16, np.int16), rate, 8000).astype(np.int16).tobytes()

    async def roundtrip():
        u = await pcm_to_ulaw(pcm16, 8000, 8000, create_stream_resampler())
        return await ulaw_to_pcm(u, 8000, 8000, create_stream_resampler())

    x = np.frombuffer(asyncio.run(roundtrip()), np.int16).astype(np.float32) / 32768.0
    x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(4000, np.float32)])
    return soxr.resample(x, 8000, 16000).astype(np.float32)


def safety_scores(items, monitor, heard=None):
    """Recall over must-fire items (any alert transfers the call) and false alarms over must-not-fire items."""
    fired = [(it, monitor.check(heard(it) if heard else it["text"])) for it in items]
    pos = [(it, a) for it, a in fired if it["expected"]]
    neg = [(it, a) for it, a in fired if not it["expected"]]
    return {"recall": (sum(a is not None for _, a in pos), len(pos)),
            "false_alarms": (sum(a is not None for _, a in neg), len(neg)),
            "misses": [it["text"] for it, a in pos if a is None] + [f"false alarm: {it['text']}" for it, a in neg if a]}


EXCLUDED_WORDS = {"sí", "si", "no"}  # the words the gate depends on never choose the voice that tests them


def choose_voice(real, voices):
    shared = [w for w in real if w not in EXCLUDED_WORDS]
    return min(voices, key=lambda v: sum(abs(voices[v].get(w, 0.0) - real[w]) for w in shared) / len(shared))
