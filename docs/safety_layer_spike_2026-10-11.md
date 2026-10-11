# Safety layer: zero-shot NLI spike and next-step proposal (2026-10-11)

Inputs: an external design review, `docs/literature_safety_layer_2026-10-11.md`.

## Spike: zero-shot multilingual NLI next to the rules (job 7630137, H100)

Model `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT; base `microsoft/mdeberta-v3-base`, MIT),
one English hypothesis per category written before any score was seen (`eval/nli_safety_spike.py`). The only fitted
number is the alert threshold (0.90), chosen on the blind dev half; the test half was scored once.

| Blind test (typed) | Emergencies caught | False alarms |
|---|---|---|
| Rules (milestone 3b) | 22/45 (49 %) | 3/25 |
| Zero-shot NLI alone | 27/45 (60 %) | 7/25 |
| Rules OR NLI | **34/45 (76 %)** | **9/25 (36 %)** |

Per category, rules OR NLI: pregnancy bleeding 5/5; abuse, bleeding, breathing, chest pain, emergency, stroke and
suicidal 4/5 each; **overdose 1/5**. Latency: 12 ms per turn on the GPU for all nine hypotheses in one batch. The CPU
run hung and timed out (job 7630138), so CPU latency is unmeasured.

Reading: a model generalises to unseen phrasing better than the rules do (+12 emergencies when combined), but the
threshold chosen on dev did not hold on test (dev 2/25 false alarms, test 7/25). Zero-shot NLI is a baseline, not a
production layer. Both halves of `eval/blind/` are now used up for this approach too.

## Proposal (needs the owner's decision; nothing below is built)

1. Opening notice, drafted for clinician and owner review before any change to the call:
   "This automated service is for scheduling and cannot provide emergency help. If you or someone else is in immediate
   danger or having a medical emergency, hang up and call 911." Today only the alert script mentions 911.
2. A small fine-tuned multilingual encoder (about 100-300M parameters, e.g. mDeBERTa-v3-base, MIT) with nine category outputs
   and an overall "human review needed" output, trained on locally generated synthetic turns with hard negatives
   ("my knee is hurting me", "chest X-ray results", resolved past symptoms), run as rules OR model.
3. Separate "transfer for human review" from "emergency script": a loud emergency message can endanger an abuse caller
   whose abuser is nearby (external review). Routing policy needs clinical review.
4. A new blind evaluation set built by scenario families, with bilingual reviewers. A per-category claim needs on the
   order of 200 positives per category; that is reviewer work, not a solo task.
5. Ship bar: no emergency-protection claim until the recall lower bound and the transfer load are measured on the
   phone path and the production hardware is chosen.
