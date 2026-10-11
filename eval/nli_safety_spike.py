"""Spike (2026-10-11): does a zero-shot multilingual NLI model catch emergencies the rules miss?

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT). One English hypothesis per category, written
before any score was seen; the caller's turn (es or en) is the premise. The only fitted number is the alert threshold,
chosen on the blind DEV half (highest dev recall with at most 2 dev false alarms). The TEST half is then scored once.
Measurement only: nothing here is used by the live agent.

usage: python eval/nli_safety_spike.py   (needs torch + transformers; HF_HOME pointing at the downloaded model)
"""
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from puente_voice.safety import SafetyMonitor

MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HYPOTHESES = {
    "suicidal": "The speaker wants to die or wants to hurt or kill themselves.",
    "breathing": "Someone cannot breathe or is choking right now.",
    "chest_pain": "Someone has chest pain or chest pressure right now.",
    "stroke": "Someone has signs of a stroke, such as a drooping face, a weak or numb arm, or slurred speech.",
    "overdose": "Someone took too many pills, too much medicine, drugs or a poison.",
    "pregnancy_bleeding": "A pregnant woman is bleeding.",
    "bleeding": "Someone is bleeding heavily.",
    "abuse": "Someone is being hit, threatened or abused by another person.",
    "emergency": "The speaker says this is an emergency or asks for an ambulance or urgent help.",
}
HERE = Path(__file__).resolve().parent


def load(split):
    return yaml.safe_load(open(HERE / f"blind/safety_{split}.yaml", encoding="utf-8"))


class NLI:
    def __init__(self, device):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
        self.device = device
        self.entail = self.model.config.label2id.get("entailment", 0)

    @torch.no_grad()
    def scores(self, text):
        """Entailment probability of every category hypothesis for one caller turn (one batch)."""
        hyps = list(HYPOTHESES.values())
        enc = self.tok([text] * len(hyps), hyps, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return torch.softmax(self.model(**enc).logits, -1)[:, self.entail].cpu().numpy()


def evaluate(items, max_scores, thr, rules):
    pos = [i for i, it in enumerate(items) if it["expected"]]
    neg = [i for i, it in enumerate(items) if not it["expected"]]
    nli = max_scores >= thr
    rule = np.array([rules.check(it["text"]) is not None for it in items])
    out = {}
    for name, fired in (("rules", rule), ("nli", nli), ("rules OR nli", rule | nli)):
        out[name] = (int(fired[pos].sum()), len(pos), int(fired[neg].sum()), len(neg))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nli, rules = NLI(device), SafetyMonitor()
    dev, test = load("dev"), load("test")
    s_dev = np.array([nli.scores(it["text"]) for it in dev]).max(1)
    neg = np.array([not it["expected"] for it in dev])
    # lowest threshold that keeps NLI-alone dev false alarms at or below 2 (recall first)
    thr = next(t for t in np.round(np.arange(0.05, 0.99, 0.01), 2) if (s_dev[neg] >= t).sum() <= 2)
    print(f"device {device} | threshold {thr} (chosen on dev)")
    for name, (tp, p, fa, n) in evaluate(dev, s_dev, thr, rules).items():
        print(f"DEV  {name:13} recall {tp}/{p}  false alarms {fa}/{n}")
    s_test = np.array([nli.scores(it["text"]) for it in test]).max(1)
    for name, (tp, p, fa, n) in evaluate(test, s_test, thr, rules).items():
        print(f"TEST {name:13} recall {tp}/{p}  false alarms {fa}/{n}")
    cats = sorted({it["expected"] for it in test if it["expected"]})
    rule_hits = [rules.check(it["text"]) is not None for it in test]
    print("TEST per category (rules OR nli): " + ", ".join(
        f"{c} {sum(1 for i, it in enumerate(test) if it['expected'] == c and (rule_hits[i] or s_test[i] >= thr))}"
        f"/{sum(1 for it in test if it['expected'] == c)}" for c in cats))
    lat = []
    for it in dev[:40]:
        t = time.perf_counter(); nli.scores(it["text"]); lat.append(time.perf_counter() - t)
    print(f"latency per turn ({device}, 9 hypotheses batched): mean {np.mean(lat)*1000:.0f} ms, "
          f"p95 {np.percentile(lat, 95)*1000:.0f} ms")


if __name__ == "__main__":
    main()
