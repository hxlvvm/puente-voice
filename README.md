# Puente Voice

Open-source phone agent that answers a US clinic's calls in Spanish or English and books appointments through FHIR.
Milestone 1 is the text-only core: call flow, scheduling, safety monitor, speech templates, audit log, simulator.
Milestone 2 puts it on the phone line: a Twilio Media Streams WebSocket server running one Pipecat pipeline per call
(Silero VAD, faster-whisper large-v3-turbo, the call flow, Kokoro TTS) and a simulated caller that dials it.
Milestones 3a and 3b measure how well it understands callers on phrasings held out from the rules.
Milestone 3c adds a safety classifier after the rules. On a fresh blind test set, rules OR classifier catch 124/144
emergencies, but suicidal statements only 4/16. **It is not ready for real callers.**
Milestone 4 is a demo deployment, not for production: Docker Compose with Asterisk 22, callable from a softphone
(`docs/deployment.md`).

Not a medical device. It never triages or diagnoses. Tests and the simulator use synthetic data only.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/python -m puente_voice.sim.text_sim scenarios
```

Audio calls (one GPU; models in a local directory, see `jobs/audio_sim.sbatch`):

```bash
.venv/bin/pip install -e ".[dev,audio]"
sbatch --array=0-3 jobs/audio_sim.sbatch <models-dir>
.venv/bin/python -m puente_voice.sim.audio_sim --merge runs/audio/calls-<array-job-id>-s*of4.jsonl
```

## Status (2026-10-11)

Simulated phone calls, 8 kHz μ-law, synthetic caller voices (Kokoro), H100. Kokoro runs on the GPU through onnxruntime-gpu 1.24.4, which the
job puts on `PYTHONPATH` (it would replace the CPU build that `.[audio]` installs); with CPU-only Kokoro, dynamic
prompts (date read-back, slot list) take about 2.4 s to start. WER on synthetic speech is a lower bound
for real callers, not a field number; it counts only the turns the caller actually spoke.

| Run | Scenarios | Wrong bookings | Latency p50 / p95 | WER es / en |
|---|---|---|---|---|
| Scripts as written (job 7584696, after the final review fixes) | 5 / 18 | 0 | 0.63 s / 1.76 s | 11.5 % / 15.0 % |
| One-word "sí" said as "sí, correcto" (job 7612548, diagnostic, same code) | 17 / 18 | 0 | 1.03 s / 1.78 s | 0.6 % / 15.0 % |
| Scripts as written, caller voice ef_dora (job 7619299, milestone 3a) | 5 / 18 | 0 | 0.72 s / 1.78 s | 12.3 % / 15.0 % |
| Milestone 3c: 46 s notice, safety classifier on (job 7637357, A100 nodes) | 5 / 18 | 0 | 0.97 s / 2.32 s | 11.5 % / 15.0 % |

The milestone 3c run passes and fails the same 18 scenarios, each with the same transfer reason, as job 7619299.
The classifier caused no false transfers, and the emergency call still gets the emergency script. Its higher latency
is not attributed to the classifier: the run landed on A100 nodes, and the earlier runs used H100s.

The test caller's Spanish voice (ef_dora) came from `eval/voice_match.py` (job 7619172), but no Kokoro Spanish voice
was distinguishable on real-vs-synthetic agreement: one take of "vale" decided it, and ef_dora is also the agent's
voice. Its one-word "sí" fails like the others.

Gates: latency p95 ≤ 2.5 s **met**; wrong bookings 0 **met**; emergency script on emergency calls **met**; happy-path
booking ≥ 90 % **not met** in simulation. The calls that fail answer with a one-word synthetic "sí" or "Hola", which
Whisper hears as "See you." / "Go lag.". The bake-off below points to the synthetic voice for "Hola" (real recordings
pass 9/10, Kokoro's 1/3), but **"sí" itself is unmeasured**: no open set used here has an isolated real "sí", so the
happy-path gate stays unexplained until it is measured.

### Speech recognition on 8 kHz phone audio (2026-10-11)

Same phone codec and segment shape as the live pipeline; forced Spanish for Whisper (Parakeet cannot be forced).

| Test set | Whisper large-v3-turbo (H100) | Parakeet-TDT-0.6B-v3 int8 (CPU, 4 threads) |
|---|---|---|
| Real sentences, FLEURS es_419, 80 utterances | **3.0 % WER**, 0.13 s each | 5.4 % WER, 0.95 s mean / 1.5 s p95 |
| Real short words, MSWC, 164 clips | **131 / 164** (hola 9/10, claro 24/25, gracias 23/25) | 100 / 164 (drifts into English: "one", "Well") |
| Synthetic Kokoro "sí" / "Hola" | 0/3 / 1/3 | fails too |
| Line noise and a cough | empty (correct) | empty (correct) |

Whisper stays the recogniser on the GPU (job 7618130). MSWC words are cut from longer sentences ("Claro que…"), so they stand in for, but are not, a
standalone "sí". Common Voice's single-word recordings include Spanish "sí" and "no" and would settle it, but need
Mozilla's terms accepted first.

CPU rows (4 threads, int8, job 7618130), measured on 20 sentences and 33 short words: Whisper small 4.3 % WER, 1.6 s
mean / 1.9 s p95 per sentence but 3.9 s p95 on short clips (over the 2.5 s gate); Whisper medium 2.2 % at 4.8 s / 5.5 s,
and it turned a synthetic "no" into the caption "Nos vemos en el próximo video"; Parakeet on the same subset 5.1 %
(2 threads). The hardware choice, and with it the CPU recogniser, is milestone 4's.

### Second safety layer: a classifier ORed with the rules (milestone 3c, scored once, 2026-10-12)

This is an agent-written development measurement, not a release claim. The caller-facing scripts below are approved by
the project owner; **clinician review is pending**.

- **Model.** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT), fine-tuned as a multi-label classifier
  (any emergency + 9 categories).
- **Training data.** About 15 k synthetic turns, written locally by Qwen2.5-7B-Instruct (Apache-2.0). No PHI.
- **Runtime.** Exported to ONNX and run with onnxruntime; there is no torch in the product.
- **When it runs.** Only when the rules stay silent.
- **What a classifier alert does.** The call goes to staff with a quieter script ("Para ayudarle mejor, le comunico
  ahora con una persona del equipo…"). Abuse gets a discreet script. Every call now opens with a 911 notice.
- **If the classifier fails.** The call continues on the rules and the audit log records `safety_model_error`.
- **Alert threshold.** The lowest value that fires on none of the 296 ordinary answers in the meaning corpora and on
  at most 3 % of the calibration nulls. That value is 0.99.

`eval/blind2/` was written by an agent that was not allowed to read the repo. Each half has 144 emergencies (16 in
each of 9 categories) and 160 non-emergencies. Every choice used the dev half only:

- Four training-data variants were tried. The plain generated data won: NLI cleaning, and hand-written suicidal
  examples with and without ordinary counter-examples, all lowered dev recall.
- The suicidal rules were extended from dev misses.

The test half was then scored once, typed (job 7637247) and over the phone line (job 7637248). One test item
nearly duplicates an item of the older `eval/blind/` set, which is part of the threshold calibration.

| Blind2 test | Rules alone | Classifier alone | **Rules OR classifier** | Phone line, rules OR classifier |
|---|---|---|---|---|
| Emergencies caught | 39/144 (27 %, CI 20–35 %) | 118/144 (82 %, CI 75–87 %) | **124/144 (86 %, CI 80–91 %)** | 325/378 (86 %) |
| False alarms on non-emergencies | 3/160 | 1/160 | **4/160 (2 %, CI 1–6 %)** | 9/416 (2 %) |

Per category, typed, rules OR classifier: bleeding 16/16, chest pain 16/16, emergency/ambulance 16/16, stroke 16/16,
breathing 15/16, bleeding in pregnancy 15/16, abuse 14/16, overdose 12/16, **suicidal 4/16**. All 4 false alarms
are "resolved" turns (a past symptom that is fine now): 0/60 in-flow answers, 0/50 near-misses ("casi me da un infarto
cuando vi la factura"), 0/25 ordinary requests. The phone line uses 3 Spanish and 2 English voices per phrase, so its
interval would be optimistic and is not shown.

| Gate (written before training) | Result |
|---|---|
| Recall, rules OR classifier, above rules alone | **met**: 124/144 vs 39/144 |
| No alarm on the 296 ordinary answers in the meaning corpora | **met**: 0/296 |
| Text-sim scenarios, no false transfers | **met**: 14/14 |
| ONNX vs torch, max probability difference < 1e-3 | **met**: 9e-6 CPU, 6e-6 GPU (with TF32 off) |
| Latency per turn | CPU (4 threads) p50 28 ms / p95 36 ms; GPU p50/p95 3 ms |

**Main finding: the classifier carries emergency detection, and the rules are now the floor.**

**Suicidal risk is the weak category: 4/16 on test.** The rules extended from dev misses took dev to 14/16 and did not
carry over.

Reading the test misses after scoring shows two kinds:
- **Plans and methods, with no suicide word:** a rope, a running car in a closed garage, a goodbye text, pills
  "to take them all tonight", jumping in front of a train. Rules cannot list these.
- **Variants of phrases the rules already know, in a grammatical form they did not cover:** "me voy a quitar la vida"
  (pronoun before the verb), "no quiero estar viva", "sleep and never wake up", "feel like ending it".

The second kind was fixed after scoring, with self-written tests. On re-reading the same test items, rules alone
would now catch 8/16 suicidal ones. That number comes from reading the test, so it is not a measurement. The scored
number above stands.

The classifier did not learn this category. The generator turns suicidal statements into vague worry, and
hand-written examples made it worse: the model learned "my husband" or "no quiero" as cues, ordinary scheduling turns
scored 0.99 or higher, and recall in other categories fell.

**This system must not be described as detecting suicide risk.** Fixing it needs clinician-written or
clinician-reviewed phrasing, and a fresh blind set: this test half is now used up.

The opening notice now starts with a short 911 sentence in each language, the usual opening of a medical office's
phone line ("If this is an emergency or you are in danger, hang up and call 911."). As Kokoro speaks it (job 7637506),
the Spanish caller hears 911 by second 4 and the English caller by second 9. The whole notice runs 35 s.

The first 3c wording put the 911 sentence at the end of each language's paragraph. That notice ran 46 s, with 911 at
about second 20 and second 42; milestone 2's had no 911 sentence and ran about 21 s.

The caller is still not heard during the notice. Listening for emergencies during it was not done for two reasons:
- transcribing before the recording notice has played is a consent-law question (**[COUNSEL]**);
- the notice itself says "911" and "emergencia", so echo on a speakerphone could set off a false transfer.

### Does the call flow understand callers? Blind held-out set (milestone 3b, scored once, 2026-10-11)

`eval/blind/` was written by an agent that was not allowed to read the code, the tests or the earlier corpora. Rules
changed only on misses from its dev half. The test half was scored once, after the last rule change, and is reported
unedited. Typed counts are distinct phrases with 95 % Wilson intervals. "Phone line" = each phrase spoken by 3 Spanish
/ 2 English Kokoro voices through the 8 kHz codec and the live `transcribe()` (jobs 7629344, 7629345); those
renderings are not independent, so the interval on them is optimistic.

| Caller answer (test) | Typed, before 3b | **Typed, after 3b** | Phone line, after 3b |
|---|---|---|---|
| Date of birth (incl. dates in words, two-digit years) | 7/16 | **16/16** (CI 81–100 %) | 39/42 |
| Slot choice | 12/16 | **16/16** (CI 81–100 %) | 39/42 |
| Yes / no | 16/16 | **16/16** (CI 81–100 %) | 42/42 |
| Language | 12/12 | **12/12** (CI 76–100 %) | 30/30 |
| Asks for a person | 13/16 | **14/16** (CI 64–97 %) | 37/42 |
| Name | 3/16 | **12/16** (CI 51–90 %) | **17/42** |
| Visit type | 8/16 | **10/16** (CI 39–82 %) | 26/42 |

| Safety monitor (test) | Typed, before 3b | **Typed, after 3b** | Phone line, after 3b |
|---|---|---|---|
| Emergencies caught | 14/45 | **22/45 (49 %, CI 35–63 %)** | 58/117 |
| False alarms on ordinary or resolved speech | 4/25 | **3/25 (CI 4–30 %)** | 9/65 |

Per category, typed (test): suicidal 4/5, emergency/ambulance 4/5, breathing 3/5, chest pain 3/5, bleeding in
pregnancy 3/5, stroke 2/5, bleeding 2/5, overdose 1/5, **abuse 0/5**.

**The main finding: rules generalise for understanding answers, but not for emergencies.** On the dev half the
broadened safety rules caught 45/45, because they were fitted to it. On unseen phrasing they catch about half. A rule list
cannot reach an emergency recall that is safe to ship. The next step was a model-based safety layer, with the rules kept as the floor, measured on a fresh
blind set: milestone 3c, above. The near-miss guards in `tests/test_safety_blind_dev.py` (for
example "no puedo hablar bien inglés" is not a stroke) are self-written, not blind.

Names lose most on the phone line. Whisper mishears the filler before the name ("Este" → "E", "mire" → "Mayre"), and
courtesy tails the dev half never had ("mucho gusto") stay attached. A wrong name cannot book for the wrong person:
the patient is matched on date of birth plus every spoken name token, so a stray word gives no match. The flow asks for
the name once more and then hands the caller to staff, and the agent creates no patient records. The cost is a
transferred call, not a wrong booking. Visit type misses are phrasings with no synonym in `config/clinic.example.yaml`. Any further
rule change needs a new blind set; this test half is now used up.

### Earlier measurement (milestone 3a, kept as history)

Frozen corpora in `eval/` (written before any score was seen; dev/test split; no rule was tuned on test items).
**These test numbers are optimistic:** the corpus author wrote them with the rule lists in hand, so about a third of
the meaning items and every safety item the monitor caught mirror a rule. The next held-out set is written blind to
`interpret.py` and `safety.py`. Typed counts are distinct phrases with 95 % Wilson intervals. "Phone line" = each
phrase spoken by 3 Spanish / 2 English Kokoro voices through the 8 kHz codec and Whisper (jobs 7619300, 7619301);
those renderings are not independent, so they carry no interval.

| Caller answer | Typed text | Phone line |
|---|---|---|
| Slot choice | 6/6 (CI 61–100 %) | 16/16 renderings |
| Date of birth | 3/6 (CI 19–81 %; dates in words fail) | 15/16 (Whisper writes digits) |
| Name | 4/6 (CI 30–90 %; fillers "Este…", "Uh" kept) | 13/16 (one phrase, three voices) |
| Yes / no | 10/11 (CI 62–98 %; "Vale" unclear) | 26/29 |
| Visit type, language, "persona" | 11/11 (CI 74–100 %) | 27/27 (language forced: see below) |

Dev-split misses not in the test numbers: "Un segundo" (wait a second) is read as option 2, "Eso es" (that's it) as
unclear, and "1 de enero del 90" (two-digit year) is not parsed. Real "vale" was heard as "¡Gracias!" in 3 of 25 MSWC
clips (job 7616461), which the live pipeline drops as a Whisper caption; a correctly heard "vale" still reads as
unclear. The phone-line runs above forced Whisper to each item's language and did not apply the live caption drop list;
the evaluation scripts now call the same `transcribe()` as the live pipeline for the next measurement.

| Safety monitor | Typed text | Phone line |
|---|---|---|
| Emergencies caught, phrasing that mirrors a rule | 10/10 | caught by every voice |
| Emergencies caught, **any other phrasing** | **0/4 (95 % CI 0–49 %)** | missed by every voice |
| False alarms on ordinary requests | 0/6 (CI 0–39 %) | 0/6 phrases |

The safety monitor misses "Tengo un dolor muy fuerte en el pecho", "Half my face is numb", "The bleeding won't stop" and
"My husband hits me": it recognises emergencies only in the words its rules list, and speech recognition neither added
nor rescued a miss. This is the most important finding of milestone 3a and the first
fix of the next step (broaden the rules on new development phrasings, then re-measure on a fresh held-out set).

Evaluation data: FLEURS (Google, CC-BY-4.0) and the Multilingual Spoken Words Corpus (MLCommons, CC-BY-4.0, derived
from Common Voice). Neither is redistributed here.

Design: `docs/design.md`.

## Licence and third-party components

Puente Voice is Apache-2.0 (`LICENSE`). None of the components below is vendored or redistributed here; each is
installed or downloaded separately under its own licence.

| Component | Use | Licence |
|---|---|---|
| Pipecat | call pipeline | BSD-2-Clause |
| faster-whisper; Whisper large-v3-turbo and small weights | speech recognition | MIT |
| Kokoro-82M (via kokoro-onnx) | speech synthesis | Apache-2.0 (kokoro-onnx: MIT) |
| Silero VAD | voice activity detection | MIT |
| mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | base of the safety classifier (weights not published) | MIT |
| Qwen2.5-7B-Instruct | wrote the synthetic training turns (not shipped) | Apache-2.0 |
| HAPI FHIR | FHIR test server | Apache-2.0 |
| Asterisk 22 (`andrius/asterisk` image) | demo telephony, run as its own container; only config files are in this repo | GPL-2.0 |
