"""Second safety layer: a fine-tuned multilingual encoder (train/safety_train.py) run with onnxruntime.

It only adds alerts next to the rule monitor (the flow ORs them); it never suppresses a rule alert. Loads model.onnx,
tokenizer.json and safety_model.json (labels, threshold, max_len) from one directory; no torch needed.
"""
import json
import os
from pathlib import Path

import numpy as np

from puente_voice.safety import Alert


def decide(probs, labels, threshold):
    """Alert when p(any) >= threshold; the alert's kind is the most likely category."""
    if probs[0] < threshold:
        return None
    return Alert(labels[1 + int(np.argmax(probs[1:]))], "model")


class SafetyModel:
    def __init__(self, model_dir, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        cfg = json.loads((d / "safety_model.json").read_text())
        self.labels, self.threshold = cfg["labels"], cfg["threshold"]
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(cfg["max_len"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = len(os.sched_getaffinity(0))   # default = every core on the node: oversubscribes a job
        # TF32 off: with it on, GPU probabilities drift ~2e-3 from the fp32 export, enough to move a 0.99 threshold
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}) if p == "CUDAExecutionProvider" else p for p in providers]
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=providers)

    def probs(self, text):
        enc = self.tok.encode(text or "")
        ids = np.array([enc.ids], np.int64)
        logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        return 1.0 / (1.0 + np.exp(-logits))

    def check(self, text):
        return decide(self.probs(text), self.labels, self.threshold)
