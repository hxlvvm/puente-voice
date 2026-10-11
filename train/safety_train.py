"""Fine-tune the multilingual XNLI encoder as a multi-label safety classifier, pick the threshold, export to ONNX.

Outputs (OUT_DIR): model.onnx, tokenizer.json, safety_model.json (labels, threshold, max_len), calibration.json,
parity.npz (300 turns with their torch probabilities, for the ONNX/tokenizer parity check in the product venv).

Threshold rule, fixed before training: the lowest grid value t such that p_any >= t fires on NO meaning-corpus turn
(ordinary in-flow answers) and on at most 3 % of the other calibration nulls (old blind safety sets + held-out
generated families). The new blind set (eval/blind2) is never read here.

usage: python train/safety_train.py GEN_DIR OUT_DIR
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CATS = ["suicidal", "breathing", "chest_pain", "stroke", "overdose", "pregnancy_bleeding", "bleeding", "abuse",
        "emergency"]
LABELS = ["any"] + CATS
MAX_LEN = 96
EVAL = Path(__file__).resolve().parent.parent / "eval"


def target(cat):
    y = np.zeros(len(LABELS), np.float32)
    if cat:
        y[0] = 1.0
        y[LABELS.index(cat)] = 1.0
    return y


def read_jsonl(p):
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    return [r for r in rows if not re.search(r"[\u3000-\u9fff\uac00-\ud7af]", r["text"])]   # generator drifted into CJK


def calibration_sets(gen_dir):
    old = []
    for split in ("dev", "test"):   # milestone 3b blind sets: both halves already used up, fine for calibration
        old += yaml.safe_load(open(EVAL / f"blind/safety_{split}.yaml", encoding="utf-8"))
    meaning = [it["text"] for p in (EVAL / "utterances.yaml", EVAL / "blind/meaning_dev.yaml",
                                    EVAL / "blind/meaning_test.yaml")
               for it in yaml.safe_load(open(p, encoding="utf-8"))]
    calib = read_jsonl(Path(gen_dir) / "gen_calib.jsonl")
    return ([(it["text"], it["expected"]) for it in old] + [(r["text"], r["category"]) for r in calib]), meaning


@torch.no_grad()
def predict(model, tok, texts, dev, bs=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
        out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    return np.concatenate(out)


def main(gen_dir, out_dir, epochs=3, lr=3e-5, bs=32, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), problem_type="multi_label_classification", ignore_mismatched_sizes=True,
        dtype=torch.float32,   # the checkpoint is stored in fp16 and transformers 5 keeps that by default: NaN under AdamW
        id2label=dict(enumerate(LABELS)), label2id={k: i for i, k in enumerate(LABELS)}).to(dev)
    train = read_jsonl(Path(gen_dir) / "gen_train.jsonl")
    X = [r["text"] for r in train]
    Y = np.stack([target(r["category"]) for r in train])
    print(f"train {len(X)} turns, positives {int(Y[:, 0].sum())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(X) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = order[i:i + bs]
            enc = tok([X[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            logits = model(**enc).logits
            loss = lossf(logits, torch.from_numpy(Y[idx]).to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{epochs} loss {tot / len(X):.4f}", flush=True)
        assert np.isfinite(tot), "training loss is not finite"

    cal, meaning = calibration_sets(gen_dir)
    p_cal = predict(model, tok, [t for t, _ in cal], dev)[:, 0]
    y_cal = np.array([c is not None for _, c in cal])
    p_mean = predict(model, tok, meaning, dev)[:, 0]
    thr = None
    grid = np.round(np.concatenate([np.arange(0.05, 0.99, 0.01), np.arange(0.99, 1.0, 0.001)]), 3)   # fine top end
    for t in grid:
        if (p_mean >= t).sum() == 0 and (p_cal[~y_cal] >= t).mean() <= 0.03:
            thr = float(t)
            break
    if thr is None:
        for i in np.argsort(-p_mean)[:5]:
            print(f"meaning {p_mean[i]:.4f} {meaning[i]!r}", flush=True)
        print(f"calib null FA at 0.999: {(p_cal[~y_cal] >= 0.999).mean():.4f}", flush=True)
    if thr is None:   # say what blocks every threshold before failing
        null = [t for t, c in cal if c is None]
        top = sorted(zip(p_mean, meaning), reverse=True)[:5] + sorted(zip(p_cal[~y_cal], null), reverse=True)[:10]
        print("null FA at 0.99:", float((p_cal[~y_cal] >= 0.99).mean()), "top scored nulls:", top, flush=True)
    assert thr is not None, "no threshold satisfies the rule"
    rep = {"threshold": thr, "calib_recall": float((p_cal[y_cal] >= thr).mean()), "calib_pos": int(y_cal.sum()),
           "calib_false_alarm": float((p_cal[~y_cal] >= thr).mean()), "calib_null": int((~y_cal).sum()),
           "meaning_false_alarms": int((p_mean >= thr).sum()), "meaning_turns": len(meaning)}
    print(json.dumps(rep), flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "calibration.json", "w"), indent=1)
    json.dump({"labels": LABELS, "threshold": thr, "max_len": MAX_LEN, "base": BASE}, open(out / "safety_model.json", "w"),
              indent=1)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))   # the product loads it with `tokenizers` alone
    model = model.float().cpu().eval()
    dummy = tok(["hola, necesito una cita"], return_tensors="pt")
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(out / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}}, dynamo=False)
    sample = [t for t, _ in cal][:200] + meaning[:100]
    np.savez(out / "parity.npz", texts=np.array(sample, dtype=object),
             probs=predict(model, tok, sample, "cpu"))
    print("saved", sorted(p.name for p in out.iterdir()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
