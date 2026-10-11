"""Synthetic caller turns for the safety classifier, generated locally by Qwen2.5-7B-Instruct (Apache-2.0).

Scenario families (kind x category/topic x who x register x language) are generated separately and split into
train / calibration BY FAMILY, so calibration measures phrasing the model never saw. Never reads eval/ files.

usage: python train/safety_gen.py OUT_DIR   (GPU; HF_HOME with the cached model; writes gen_{train,calib}.jsonl)
"""
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CATEGORIES = {
    "suicidal": "the person wants to die, is thinking of killing themselves, or intends to hurt themselves",
    "breathing": "someone cannot breathe, is choking, gasping, or turning blue right now",
    "chest_pain": "someone has chest pain, pressure or tightness right now, possibly spreading to the arm or jaw, "
                  "sweating or nausea",
    "stroke": "someone suddenly has a drooping face, one-sided weakness or numbness, slurred or confused speech",
    "overdose": "someone took too many pills, drugs or medicine, or a child swallowed medicine or a chemical",
    "pregnancy_bleeding": "a pregnant woman is bleeding",
    "bleeding": "someone is bleeding heavily or the bleeding will not stop",
    "abuse": "the caller (or someone they speak for) is being hit, threatened, controlled or abused by a person, "
             "possibly someone nearby",
    "emergency": "the caller says it is an emergency, asks for an ambulance or 911, or urgent help, without "
                 "naming a specific condition",
}
WHO = {"self": "the caller themselves", "child": "the caller's child", "parent": "the caller's elderly parent",
       "partner": "the caller's husband, wife or partner", "other": "a friend, neighbour or relative with the caller"}
WHO_PREG = {"self": "the caller herself", "partner": "the caller's wife or partner", "child": "the caller's daughter"}
REGISTER = {"calm": "calm and matter-of-fact, almost understated",
            "panicked": "frightened and rushed, short broken sentences",
            "rambling": "rambling, with fillers and hesitations, the key point buried mid-sentence",
            "embedded": "it comes up while they are trying to book or change an appointment",
            "indirect": "indirect: they describe what is happening without naming the condition",
            "spanglish": "Spanglish, mixing English words into Spanish as many US callers do"}
LANG = {"es": "Spanish as spoken by US Hispanic callers (Mexican, Central American, Caribbean)", "en": "US English"}
INFLOW = {"name": "their full name, when asked for it", "dob": "their date of birth, when asked, in varied ways",
          "phone": "a phone number, when asked", "yesno": "yes or no or a short confirmation to 'is that correct?'",
          "choice": "which of three offered appointment times they want", "visit": "the reason for the visit "
          "(checkup, follow-up, results, vaccine, physical)", "language": "which language they prefer",
          "repeat": "asking to repeat, to wait a moment, greetings, thanks, goodbye"}
NEARMISS = {"idiom": "everyday idioms with words like dying, killing, dead, choke, heart attack used figuratively",
            "xray": "scheduling a chest X-ray, an EKG, a cardiology or heart check-up",
            "prenatal": "booking a routine prenatal or pregnancy check-up, nothing wrong",
            "blooddraw": "a blood test, blood draw, lab work or blood pressure check",
            "aches": "a sore knee, back pain, a mild headache, a cough or a cold, nothing urgent",
            "meds": "refilling or asking about their medication, pills or prescription",
            "healing": "a cut, burn or injury that was already treated and is healing",
            "urgentform": "something 'urgent' that is only paperwork, a form, a school or work note",
            "history": "talking about a relative's past illness or death, not happening now",
            "therapy": "booking a mental-health or counselling appointment, calm, no danger"}
ORDINARY = {"hours": "opening hours, location, parking", "insurance": "insurance, payment, cost",
            "doctor": "which doctor, a male or female doctor, a Spanish-speaking doctor",
            "family": "bringing a relative along, booking for a family member", "portal": "the patient portal, "
            "results by email, a letter"}
REG_NEG = {"plain": "plain and polite", "chatty": "chatty, with fillers and small talk",
           "terse": "very short, a few words", "spanglish": "Spanglish, mixing English words into Spanish"}
N_PER_PROMPT = 8


def families():
    """(family_id, kind, label, prompt-spec) for every scenario family."""
    out = []
    for cat, desc in CATEGORIES.items():
        whos = WHO_PREG if cat == "pregnancy_bleeding" else WHO
        for (who, wdesc), (reg, rdesc), (lang, ldesc) in itertools.product(whos.items(), REGISTER.items(), LANG.items()):
            if reg == "spanglish" and lang == "en":
                continue
            spec = (f"Situation: {desc}. It concerns {wdesc}. Style: {rdesc}. Language: {ldesc}. Each line must make "
                    f"the situation clear to a careful listener, but vary the wording; do not always use the obvious "
                    f"keyword.")
            out.append((f"pos/{cat}/{who}/{reg}/{lang}", "emergency", cat, spec, lang))
    for (kind, topics) in (("inflow", INFLOW), ("nearmiss", NEARMISS), ("ordinary", ORDINARY)):
        for (t, tdesc), (reg, rdesc), (lang, ldesc) in itertools.product(topics.items(), REG_NEG.items(), LANG.items()):
            if reg == "spanglish" and lang == "en":
                continue
            spec = (f"The caller is in an ordinary, non-emergency scheduling call. Topic: {tdesc}. Style: {rdesc}. "
                    f"Language: {ldesc}. Nothing dangerous is happening now.")
            out.append((f"{kind}/{t}/{reg}/{lang}", kind, None, spec, lang))
    for cat, desc in CATEGORIES.items():
        for (reg, rdesc), (lang, ldesc) in itertools.product(REG_NEG.items(), LANG.items()):
            if reg == "spanglish" and lang == "en":
                continue
            spec = (f"The caller mentions that in the past (clearly over and resolved) or NOT now: {desc}; or they "
                    f"say clearly that they do not have it. They only want a routine appointment. Style: {rdesc}. "
                    f"Language: {ldesc}. Nothing dangerous is happening now.")
            out.append((f"resolved/{cat}/{reg}/{lang}", "resolved", None, spec, lang))
    return out


def prompt(spec):
    return [{"role": "system", "content": "You write realistic, varied caller speech for training the safety filter of "
                                          "a clinic phone system. The filter exists to route callers in crisis to "
                                          "human help, so emergency lines must state the situation plainly enough "
                                          "that it is unmistakable; do not soften or replace it with a milder "
                                          "problem. Output only a JSON array of strings."},
            {"role": "user", "content": f"Write {N_PER_PROMPT} different things a caller might say in ONE turn on "
                                        f"the phone to a clinic's automated scheduling line. {spec} Write how people "
                                        f"really talk on the phone, 3 to 35 words each. Output only a JSON array of "
                                        f"{N_PER_PROMPT} strings."}]


def parse(text):
    m = re.search(r"\[.*\]", text, re.S) or re.search(r"\[.*", text, re.S)   # the array may be cut off
    block = m.group(0) if m else text
    try:
        items = json.loads(block)
    except json.JSONDecodeError:   # trailing commas, a cut-off array: keep every complete quoted string
        items = []
        for q in re.findall(r'"((?:[^"\\]|\\.)*)"', block):
            try:
                items.append(json.loads(f'"{q}"', strict=False))   # strict=False: raw newlines inside a string
            except json.JSONDecodeError:
                continue
    return [s.strip() for s in items if isinstance(s, str) and 3 <= len(s.strip()) <= 300]


def calib_family(fid):
    return int(hashlib.sha256(fid.encode()).hexdigest(), 16) % 100 < 15   # 15 % of families held out


def main(out_dir, rounds=2, batch=48, only=None):
    random.seed(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    fams = [f for f in families() if only is None or f[2] == only]   # only: regenerate one category
    jobs = [f for f in fams for _ in range(rounds if f[1] == "emergency" else 3 * rounds)]  # false alarms decide
    seen, rows, short = set(), [], 0
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        texts = [tok.apply_chat_template(prompt(f[3]), tokenize=False, add_generation_prompt=True) for f in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=700, do_sample=True, temperature=0.9, top_p=0.95)
        for f, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            fid, kind, cat, _, lang = f
            raw = tok.decode(g, skip_special_tokens=True)
            items = parse(raw)
            if len(items) < 3:
                short += 1
                if short <= 3:
                    print("SHORT OUTPUT:", repr(raw[:400]), flush=True)
            for s in items:
                key = re.sub(r"\W+", " ", s.lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"family": fid, "kind": kind, "category": cat, "lang": lang, "text": s,
                             "split": "calib" if calib_family(fid) else "train"})
        print(f"{min(i + batch, len(jobs))}/{len(jobs)} prompts, {len(rows)} turns, {short} short", flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calib"):
        with open(out / f"gen_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: sum(1 for r in rows if r["kind"] == k) for k in ("emergency", "inflow", "nearmiss", "resolved",
                                                                "ordinary")}, "families", len(fams))


if __name__ == "__main__":
    main(sys.argv[1], only=sys.argv[2] if len(sys.argv) > 2 else None)
