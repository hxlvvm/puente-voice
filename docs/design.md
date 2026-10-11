# Puente Voice: automated Spanish/English phone agent for clinic appointment scheduling

Design spec, 2026-10-09, written before any code. Milestones 1-4 implement it; the README has the current results.

## 1. Purpose and context

**What it is.** An open-source, self-hostable phone agent that answers a US clinic's calls in Spanish or English and
books appointments into the clinic's scheduling system through FHIR, with no human representative on routine calls.

**Who it is for.** Clinics, or a service provider operating it for them, and anyone who wants to self-host it
(Apache-2.0).

**Why.** Spanish-speaking patients wait for scarce bilingual staff. v1 starts with scheduling because it is
administrative, lower-risk work. Later versions add new-patient intake, behavioral-health intake and
prior-authorization calls on the same platform.

**Built on.**
- [Puente](https://github.com/hxlvvm/puente): its clinical-Spanish pitfall lexicon, its 296-case benchmark for choosing
  the LLM, and its evidence-first design.
- A healthcare/compliance review of Puente found that its demo code is **not** safe to reuse as-is for real calls: no
  auth, a hosted free-tier model by default, an evidence check that can be fooled, and a "no emergency signs" message.
  Puente Voice is a new codebase that reuses the ideas, not that code.

**Success for v1.**
- A simulated caller on the cluster completes a Spanish booking call end to end: disclosure, identity, slot offer,
  read-back, booking. The appointment then exists on a test FHIR server, with its Provenance record.
- Every acceptance gate in section 9 passes.
- The system runs from Docker Compose on a free cloud VM, where the owner places a live test call from a softphone.

## 2. Scope

**In v1.**
- Inbound calls from existing patients, in Spanish and English.
- Choosing a visit type and an open slot, booking it, and confirming it.
- Transfer to a human on request, on failure, or on an emergency mention.
- New callers: collect name, date of birth and phone, then hand off to a human.

**Designed for, but not built in v1.**
- New-patient intake, behavioral-health intake and prior-authorization flows.
- EHR-specific adapters (athenahealth, eClinicalWorks, NextGen) using SMART on FHIR Backend Services.
- Telephony adapters for Twilio, Genesys AudioHook and Amazon Connect.
- Multiple tenants and billing; SMS reminders; outbound calls.

**Never.**
- Triage, diagnosis, or medical advice.
- Sending real patient data (PHI) to any service without a Business Associate Agreement (BAA).

## 3. Compliance constraints

These are **design inputs, not legal advice**. Items marked [COUNSEL] must be confirmed by a lawyer or compliance
officer before any real call.

- **HIPAA.** Whoever operates it for a clinic becomes a business associate of that clinic, so a BAA is needed with every
  vendor that touches PHI: host, SIP carrier, and any hosted model.
  - Free-tier model APIs are never used with PHI. Testing with synthetic patients may use any model, because
    synthetic data contains no PHI.
  - Real deployments use self-hosted models or BAA-covered endpoints (for example Vertex AI, Azure or Bedrock).
  - A formal risk analysis is required before go-live. [COUNSEL]
- **Operation from outside the US.** Contracts must allow any offshore access to PHI, which many US covered
  entities restrict. [COUNSEL]
- **Recording and AI disclosure.** Every call opens with a bilingual disclosure of the automated assistant and the
  transcription.
  - All-party-consent states apply to real-time processing.
  - Consent is logged per call.
  - Saying *persona* / *representative* routes the caller to a human. [COUNSEL]
- **Language access (ACA Section 1557, 45 CFR 92.201 and 92.210).** The agent serves patients directly in Spanish.
  The clinic must still offer a qualified human interpreter on request. How the rule applies to AI agents is
  [COUNSEL].
- **FDA boundary.** Scheduling is administrative. The agent never triages or diagnoses.
  - Emergency handling is worded as "you mentioned X; if this is an emergency call 911", with no clinical
    judgement.
  - Product claims never say "detects emergencies". [COUNSEL]

## 4. Architecture

```
caller ──phone──▶ Telephony adapter ──audio──▶ Speech-to-text ──text──▶ Conversation manager ──text──▶ Text-to-speech ──▶ caller
                  (Asterisk v1)                (self-hosted)            (LLM + tools)                    (self-hosted)
                                                   │                        ├──▶ Scheduling (FHIR)
                                                   └──▶ Safety monitor ─────┴──▶ emergency script + transfer
                                         every step ──▶ audit log (IDs only); transcripts encrypted
```

The orchestrator is **Pipecat** (BSD-2-Clause, v1.12.0 checked 2026-10-09). It is a single Python process that handles
voice activity detection, turn-taking and barge-in, and it already ships serializers for Twilio and Genesys.

### Units and interfaces

Each unit can be tested on its own.

| Unit | Responsibility | Interface |
|---|---|---|
| `telephony/asterisk` | Asterisk 22 LTS: separate per-leg audio via a snoop channel → externalMedia (chan_websocket or AudioSocket, slin16) → Pipecat frames; transfer via ARI | Pipecat transport/serializer |
| `flow/` | Deterministic call state machine (section 5); owns which step may run next | `Flow.on_event(event) -> Action` |
| `agent/` | LLM turn handling: phrases replies and extracts slot values; acts only through tools with JSON-schema-constrained calls | tools: `find_patient`, `search_slots`, `book_appointment`, `transfer_to_human` |
| `safety/` | Rule-based bilingual phrase detector on every partial transcript, independent of the LLM; per-clinic lists | `Safety.check(text, lang) -> Alert or None` |
| `scheduling/` | FHIR R4 client: patient match, Slot search, Appointment create, Provenance | `Scheduler` protocol; `FhirScheduler` (HAPI for tests); EHR adapters later |
| `speech/` | Speech-to-text and text-to-speech services behind Pipecat services; models chosen by measurement (section 8) | Pipecat STT/TTS services |
| `render/` | Fixed templates that speak dates, times, names and confirmations in es/en (no LLM wording) | `say_slot(slot, lang)` and similar |
| `audit/` | Append-only event log (IDs only), encrypted transcript store, retention job | `audit.event(call_id, kind, **ids)` |
| `sim/` | Simulated callers: scripted personas → TTS → 8 kHz μ-law → pjsua into Asterisk; scenario runner and metrics | CLI + scenario YAML files |

## 5. Call flow

```
DISCLOSE ─▶ LANGUAGE ─▶ IDENTIFY ─▶ VISIT ─▶ OFFER ─▶ READBACK ─▶ BOOK ─▶ CONFIRM ─▶ END
    any state ──"persona"/2 failed turns/error──▶ TRANSFER        any state ──safety alert──▶ EMERGENCY ─▶ TRANSFER
```

1. **DISCLOSE.** Play the fixed bilingual disclosure first, before any audio is transcribed for content, and log it.
2. **LANGUAGE.** Detect the language from the first utterance; the caller can switch at any time.
3. **IDENTIFY.** Collect the full name and date of birth, and read the date of birth back.
   - Proceed only on **exactly one** FHIR match on name + DOB. Zero or several matches go to TRANSFER.
   - Nothing about the patient's records is spoken before this match.
4. **VISIT.** Ask for the visit type (clinic-configured list) and any preferred provider or day.
5. **OFFER.** Offer up to 3 open slots from `search_slots`. Slots are spoken with `render/` templates in the clinic's
   time zone.
6. **READBACK.** For example: "Martes 14 de octubre, 10:30, con la Dra. García. ¿La reservo?"
   - Only a clear yes moves on.
   - An unclear answer (for example "creo que sí") gets one repeat, then TRANSFER.
7. **BOOK.** Call `book_appointment`, at most once per call (idempotency key = call ID), and only for a slot offered in
   this call and the patient ID locked in IDENTIFY.
8. **CONFIRM.** Repeat the booked details and say goodbye.

**New callers.** IDENTIFY finds no match and the caller says they are new. The agent collects name, DOB and phone, then
goes to TRANSFER (or creates a callback request if no human is available).

**Latency budget.** Measured from the end of caller speech to the first agent audio: **≤1.5 s at p50, ≤2.5 s at p95**.

## 6. Data

**Per call:** call ID, clinic ID, language, disclosure timestamp, matched patient ID, states visited, slots offered,
appointment ID, each transfer with its reason, timing for each turn, and the model/prompt/lexicon versions.

**Transcripts.** Encrypted at rest, deleted automatically after a configurable TTL (default 30 days). [COUNSEL: whether
transcripts are part of the designated record set.]

**FHIR writes.** One `Appointment`, plus a `Provenance` record that names the agent version and the call ID and states
that the patient confirmed by read-back.

**Logs.** Application logs carry IDs only: never transcript text, names, dates of birth or audio. Inference-server
prompt logging is turned off.

## 7. Safety and failure handling

**Emergency monitor.**
- Bilingual phrase rules cover chest pain, trouble breathing, stroke signs, suicidal thoughts, overdose, heavy
  bleeding, bleeding in pregnancy and abuse.
- It runs on partial transcripts; negation is handled, but when in doubt it fires.
- The LLM can raise a flag as a second layer, never as the only one.
- On a hit:
  - Stop the flow and play the fixed script: "If this is an emergency, hang up and call 911. If you are thinking about
    hurting yourself, call or text 988."
  - Transfer to clinic staff and log the event.
- The agent never says that a caller has no emergency.

**Failure responses.**

| Failure | Response |
|---|---|
| Speech not understood | Re-ask up to 2 times, then TRANSFER |
| LLM slow (over 3 s) or tool call invalid | "One moment", retry once, then TRANSFER |
| Slot taken before booking | Search again, offer new slots |
| Scheduling backend down | Apologise, create a callback request or TRANSFER |
| TTS fails | Pre-recorded fallback prompts |
| Call drops | No booking unless READBACK was confirmed; booking is idempotent |
| Kill switch (config) | All calls go straight to humans |

**LLM guardrails.**
- Tools validate every argument against call state.
- Caller speech is treated as data, never as instructions.
- The flow, not the LLM, decides when disclosure, read-back and booking happen.

**Security.**
- Every hop runs over TLS; SIP over TLS with SRTP in production.
- API keys in v1, then OIDC single sign-on.
- Error responses never echo caller content.
- Dependency and model versions are pinned.

## 8. Models, chosen by measurement

**Candidates (licences checked 2026-10-09).**

| Layer | Candidates |
|---|---|
| Speech-to-text | Voxtral Mini 4B Realtime (Apache-2.0, GPU via vLLM); Parakeet-TDT-0.6B-v3 (CC-BY-4.0, CPU via sherpa-onnx); faster-whisper large-v3-turbo (MIT) as the baseline |
| LLM (via vLLM with json_schema decoding) | Ministral-3-8B/14B, Qwen3.5-9B, Gemma-4-12B, gpt-oss-20b (all Apache-2.0) |
| Text-to-speech | Kokoro-82M (Apache-2.0); Qwen3-TTS (Apache-2.0); Piper voices with clean licences only (davefx CC0, claude Apache-2.0, ald Unlicense; the runtime is GPL) |

**Selection criteria.**
- Speech-to-text: WER on **8 kHz μ-law Spanish** (no published numbers exist; older models score 17–25 %).
- LLM: Puente's pitfall benchmark plus scheduling-scenario success.
- All models: latency on the target hardware.

**Licence check.** Licences are re-checked on every upgrade, because mod_audio_stream and Piper changed licence
recently.

## 9. Testing and acceptance

**Unit tests (CI).**
- Flow transitions, including that READBACK cannot be skipped.
- es/en templates.
- Tool guards.
- Safety detector positives and negations.
- FHIR client against HAPI FHIR.

**Simulated calls (cluster).** A scripted caller speaks over the Twilio Media Streams protocol to the real speech
pipeline on one GPU node, with no external network (Asterisk joins in milestone 4). Scenarios
are YAML files, each with a required outcome:
- happy path (es and en);
- wrong DOB; no match; two matches;
- slot taken mid-call;
- barge-in;
- "persona";
- emergency mentioned mid-booking;
- Spanglish;
- Puente false friends;
- hesitant yes;
- silence;
- background noise.

**Gates before any live human test call.**

| Metric | Gate |
|---|---|
| Wrong bookings | 0 |
| Emergency script on emergency scenarios | 100 % |
| Happy-path booking success | ≥ 90 % |
| Latency p95 | ≤ 2.5 s |
| Spanish 8 kHz WER | measured and reported |

**Gates before any real patient call** (owner plus clinic, later):
- native-speaker review of every Spanish prompt and phrase list;
- legal sign-off on the [COUNSEL] items;
- a shadow-mode pilot;
- for the emergency monitor, at least 60 positive examples per language with zero misses.

## 10. Milestones

1. **Skeleton.** Repo, flow, scheduling against HAPI FHIR, render templates, safety detector, a text-only simulator, CI.
2. **Audio.** Pipecat pipeline and the speech services. Simulated calls on the cluster use the Twilio Media Streams
   protocol (8 kHz μ-law over a WebSocket; Pipecat's built-in serializer). Includes latency measurement. (Owner
   decision 2026-10-09: this is free and local and exercises the adapter US clients use; the Asterisk adapter moved to
   milestone 4.)
3. **Selection.** Model selection and the safety-monitor evaluation; results tables in the README.
4. **Deployment.** Docker Compose on a free cloud VM, including the Asterisk 22 adapter (per-leg audio via
   externalMedia); live softphone test by the owner; documentation.

## 11. Risks

1. **Telephone-quality Spanish speech-to-text is unmeasured.** It may need the larger GPU model or a fine-tune.
2. **Code-switching and false friends.** The 7B model scored 54.6 % on Puente's pitfall benchmark.
3. **Real-EHR write access** depends on vendor and site approval (SMART registration), which is outside our control.
4. **Licence drift** in the speech and telephony components.
5. **Recording-consent and AI-disclosure laws** vary by state and are changing.
6. **Simulated callers may be easier than real ones.** Live role-play and later shadow pilots are required before
   claims.

## 12. Open items for later specs

- Intake, behavioral-health and prior-authorization flows.
- EHR adapters.
- Multi-tenant hosting and billing.
- Choosing the hosted model provider under BAA for production.
