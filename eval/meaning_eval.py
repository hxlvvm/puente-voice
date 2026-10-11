"""Meaning-level coverage of the rule interpreter.  python eval/meaning_eval.py [--split dev|test]"""
import argparse
import sys
from pathlib import Path

from puente_voice.evaluate import load_set, report, score
from puente_voice.interpret import RuleInterpreter
from puente_voice.sim import demo_data

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--corpus", default=str(HERE / "utterances.yaml"), help="labelled YAML set to score")
    ap.add_argument("--asr", metavar="MODELS_DIR", help="speak every item with Kokoro voices through the phone line")
    a = ap.parse_args(argv)
    items = load_set(a.corpus, a.split)
    ip = RuleInterpreter(demo_data.CLINIC)
    if not a.asr:
        print("\n".join(report(score(items, ip))))
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
            spoken.append({**it, "text": f"[{v}] {it['text']}", "heard": heard})
    print("\n".join(report(score(spoken, ip, heard=lambda it: it["heard"]))))


if __name__ == "__main__":
    sys.exit(main())
