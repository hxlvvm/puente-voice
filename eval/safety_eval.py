"""Safety monitor recall and false alarms.

python eval/safety_eval.py [--split dev|test] [--asr MODELS_DIR] [--model SAFETY_MODEL_DIR]
"""
import argparse
import sys
from pathlib import Path

from puente_voice.evaluate import load_set, safety_scores, wilson
from puente_voice.safety import SafetyMonitor

HERE = Path(__file__).resolve().parent


def show(s):
    for key in ("recall", "false_alarms"):
        k, n = s[key]
        lo, hi = wilson(k, n)
        print(f"{key:12} {k:3}/{n:<3} {k / n if n else 0:6.1%}  (95 % CI {lo:.0%}-{hi:.0%})")
    for m in s["misses"]:
        print(f"             miss: {m}")


class Either:
    """Rules first, then the safety model: the same OR the call flow applies."""
    def __init__(self, *monitors):
        self.monitors = monitors

    def check(self, text):
        return next((a for m in self.monitors if (a := m.check(text))), None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--corpus", default=str(HERE / "safety_set.yaml"), help="labelled YAML set to score")
    ap.add_argument("--asr", metavar="MODELS_DIR", help="speak every item with Kokoro voices through the phone line")
    ap.add_argument("--model", metavar="SAFETY_MODEL_DIR", help="OR the rules with the safety classifier (CPU)")
    a = ap.parse_args(argv)
    items = load_set(a.corpus, a.split)
    if a.model:
        from puente_voice.safety_model import SafetyModel
        monitor = Either(SafetyMonitor(), SafetyModel(a.model))
    else:
        monitor = SafetyMonitor()
    if not a.asr:
        show(safety_scores(items, monitor))
        return 0
    from faster_whisper import WhisperModel

    from puente_voice.evaluate import phone_segment
    from puente_voice.voice.speech import load_kokoro, synth, transcribe
    voices = {"es": ["em_alex", "em_santa", "ef_dora"], "en": ["am_michael", "af_heart"]}
    k = load_kokoro(f"{a.asr}/kokoro", 4)
    m = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
    spoken = []
    for it in items:
        for v in voices[it["lang"]]:
            # the live filters; the call language is unknown until the caller has chosen it
            _, heard = transcribe(m, phone_segment(*synth(k, it["text"], it["lang"], v)),
                                  None if it.get("kind") == "language" else it["lang"])
            spoken.append({**it, "text": f"[{v}] {it['text']} (heard {heard!r})", "heard": heard})
    show(safety_scores(spoken, monitor, heard=lambda it: it["heard"]))


if __name__ == "__main__":
    sys.exit(main())
