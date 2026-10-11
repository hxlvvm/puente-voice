"""Clean generated emergency turns with the zero-shot NLI model (eval/nli_safety_spike.py).

The generator sometimes softens an emergency into an ordinary complaint ("my daughter has cold symptoms" labelled
suicidal). A generated emergency is kept only if the NLI model finds SOME emergency in it (max entailment >= KEEP).
Non-emergency turns are not filtered: the hard negatives ("this wait is killing me") are exactly what NLI
over-fires on, and they are what the classifier must learn. Optional REPLACE_DIR swaps in a regenerated category.

usage: python train/safety_clean.py GEN_DIR OUT_DIR [REPLACE_DIR CATEGORY]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from nli_safety_spike import NLI  # noqa: E402

KEEP = 0.5


def rows(d, split):
    return [json.loads(line) for line in open(Path(d) / f"gen_{split}.jsonl", encoding="utf-8")]


def main(gen_dir, out_dir, replace_dir=None, category=None):
    nli = NLI("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calib"):
        data = rows(gen_dir, split)
        if replace_dir:
            data = [r for r in data if r["category"] != category] + rows(replace_dir, split)
        kept, dropped = [], Counter()
        for r in data:
            if r["kind"] == "emergency" and float(np.max(nli.scores(r["text"]))) < KEEP:
                dropped[r["category"]] += 1
                continue
            kept.append(r)
        with open(out / f"gen_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        pos = Counter(r["category"] for r in kept if r["kind"] == "emergency")
        print(f"{split}: kept {len(kept)}/{len(data)}; dropped emergencies {dict(dropped)}; kept per category {dict(pos)}",
              flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
