"""Gates for a trained safety model (README, milestone 3c).

python eval/safety_model_eval.py MODEL_DIR [--split dev|test] [--gpu]
Prints: ONNX/tokenizer parity against parity.npz, false alarms on the meaning corpora (ordinary caller answers),
recall and false alarms on eval/blind2/safety_<split>.yaml (rules, model, rules OR model, per category and per kind),
and per-turn latency. The test split is scored once, at the end of the milestone.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from puente_voice.evaluate import wilson
from puente_voice.safety import SafetyMonitor
from puente_voice.safety_model import SafetyModel

HERE = Path(__file__).resolve().parent


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / max(n, 1):.0%}, CI {lo:.0%}-{hi:.0%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    providers = ("CUDAExecutionProvider", "CPUExecutionProvider") if a.gpu else ("CPUExecutionProvider",)
    model, rules = SafetyModel(a.model_dir, providers), SafetyMonitor()
    print("providers", model.sess.get_providers())

    par = np.load(Path(a.model_dir) / "parity.npz", allow_pickle=True)
    diff = max(float(np.abs(model.probs(t) - p).max()) for t, p in zip(par["texts"], par["probs"]))
    print(f"parity: max |onnx - torch| = {diff:.2e} over {len(par['texts'])} turns -> {'PASS' if diff < 1e-3 else 'FAIL'}")

    meaning = [it["text"] for p in (HERE / "utterances.yaml", HERE / "blind/meaning_dev.yaml", HERE / "blind/meaning_test.yaml")
               for it in yaml.safe_load(open(p, encoding="utf-8"))]
    fired = [t for t in meaning if model.check(t)]
    print(f"meaning corpora (ordinary answers): model false alarms {len(fired)}/{len(meaning)}"
          f" -> {'PASS' if not fired else 'FAIL'}")

    items = yaml.safe_load(open(HERE / f"blind2/safety_{a.split}.yaml", encoding="utf-8"))
    r = [rules.check(it["text"]) is not None for it in items]
    m = [model.check(it["text"]) is not None for it in items]
    pos = [i for i, it in enumerate(items) if it["expected"]]
    neg = [i for i, it in enumerate(items) if not it["expected"]]
    for name, f in (("rules", r), ("model", m), ("rules OR model", [x or y for x, y in zip(r, m)])):
        print(f"blind2 {a.split} {name:15} recall {rate(sum(f[i] for i in pos), len(pos))}  "
              f"false alarms {rate(sum(f[i] for i in neg), len(neg))}")
    both = [x or y for x, y in zip(r, m)]
    cats = Counter(items[i]["expected"] for i in pos)
    print("  recall by category (OR): " + ", ".join(
        f"{c} {sum(both[i] for i in pos if items[i]['expected'] == c)}/{n}" for c, n in sorted(cats.items())))
    kinds = Counter(items[i]["kind"] for i in neg)
    print("  false alarms by kind (OR): " + ", ".join(
        f"{k} {sum(both[i] for i in neg if items[i]['kind'] == k)}/{n}" for k, n in sorted(kinds.items())))

    lat = []
    for it in items[:200]:
        t = time.perf_counter()
        model.probs(it["text"])
        lat.append(time.perf_counter() - t)
    print(f"latency per turn: p50 {np.percentile(lat, 50) * 1000:.1f} ms, p95 {np.percentile(lat, 95) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
