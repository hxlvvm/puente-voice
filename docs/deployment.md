# Demo deployment (not for production)

This runs Puente Voice on one machine so you can call it from a softphone. It is a **demo**:

- the clinic, patients and calendar are synthetic (`puente_voice/sim/demo_data.py`);
- the demo clock is fixed, so the agent always offers slots on 13-15 October 2026;
- bookings live in memory and are gone on restart.

**Do not connect it to a real EHR, a real phone number or real patients.** Below is what a production deployment would
still need.

## What runs

| Service | What it does | Exposed |
|---|---|---|
| `asterisk` | Asterisk 22 (image `andrius/asterisk`, GPLv2, run unmodified). Two SIP accounts: `caller`, who dials `100`, and `staff`, who receives transfers. | SIP 5060/udp, RTP 10000-10099/udp |
| `puente` | The agent (`python -m puente_voice.serve`). It takes each call's audio from Asterisk over AudioSocket and runs the same pipeline the simulator measures: Whisper, call flow, safety check, Kokoro. | nothing, internal network only |

**Inside a call.**
1. Asterisk answers and hands the audio to the agent over AudioSocket (8 kHz, 16-bit).
2. When the agent ends or transfers the call, Asterisk asks it `GET /outcome/<call>` once.
3. On `transferred`, Asterisk rings the `staff` softphone. Otherwise it hangs up.

There is no trunk, so no extension can dial an outside number.

## Run it

You need Docker with Compose and a machine whose UDP ports 5060 and 10000-10099 your softphone can reach. A cloud VM
needs those ports open in its firewall.

```sh
cp .env.example .env               # set PUENTE_SIP_PASSWORD (letters and digits) and PUENTE_PUBLIC_IP
scripts/fetch_models.sh            # Kokoro into ./models; Whisper downloads on first start (~1.6 GB)
docker compose up --build
```

**Set up the softphone.** Use any SIP client, for example Linphone or MicroSIP:
- server `PUENTE_PUBLIC_IP`;
- user `caller`, with the password from `.env`;
- UDP transport and the G.711 μ-law codec.

Dial `100`. A second softphone registered as `staff` rings when the agent transfers.

**Try these demo identities:**
- María José Pérez García, born 15 March 1980;
- John Smith, born 4 July 1975.

**Safety classifier.**
- By default the agent runs the **rules only**, which caught 27 % of emergencies on the blind test.
- The classifier (rules OR classifier caught 86 %) is not downloadable. To use it, copy its directory to
  `./models/safety` and set `PUENTE_SAFETY_MODEL=/models/safety`.
- Either way, the agent must not be described as detecting emergencies.

**Hardware.** The image runs on CPU by default (`PUENTE_DEVICE=cpu`), with Whisper *small* in int8. The README's
speech-accuracy numbers are for Whisper large-v3-turbo on a GPU; small's accuracy on phone Spanish is **unmeasured**.
Turbo on 4 CPU cores was too slow: its transcripts arrived after the agent's 8 s silence timer, so it re-asked every
question (job 7637774).

| On 4 CPU cores (job 7637818) | On an H100 (job 7637728) |
|---|---|
| ready about 60 s after start | not measured |
| answers about 5.5 s after the caller stops | about 1.2 s |

Wait a minute after `docker compose up` before dialing.

## What was tested, and what was not

- **Tested on the cluster.**
  - The AudioSocket bridge, against the real server with a fake Asterisk (`tests/test_audiosocket.py`).
  - The dialplan, the bridge and the agent together, through the same Asterisk 22 image under Apptainer
    (`jobs/asterisk_smoke.sbatch`). Three calls: the caller asks for staff, reports chest pain, or hangs up. Each is
    checked for audio both ways, the transfer decision and the audit log. Result: **PASS** on an H100 (job 7637728)
    and on 4 CPU cores with the GPU hidden (job 7637818, the compose default). The CPU run loads
    `deploy/asterisk/pjsip.conf.in` rendered by the same `sed` line as compose.
    - The notice reaches the caller as audio.
    - "Español, por favor" sets the language on all three calls.
    - The staff request ends in `transferred` (`caller_request`).
    - Chest pain ends in `transferred` (`emergency:breathing`, since "no puedo respirar" in the same sentence fired
      first).
    - The hang-up ends with no transfer or booking.
    - The agent answers 1.2 s after the caller stops.
- **Not tested.**
  - The Docker image build and `compose.yaml` themselves: the cluster has no Docker.
  - SIP registration, NAT and RTP from a real softphone.
  - Barge-in over RTP: the bridge cannot recall audio already sent, though a test shows it is sent at real-time
    speed.

  The owner's live softphone call is that test.

## Before anything real (production)

This demo leaves out everything below. Each item needs its own work and, where marked, a lawyer.

- **Real data.** EHR integration with SMART registration and the site's approval; real clinic config.
- **Scheduling persistence.** The demo's in-memory calendar is not durable.
- **Phone line.** A carrier trunk with toll-fraud controls.
- **Encryption.**
  - In transit: SIP over TLS and SRTP.
  - At rest: transcripts (spec section 6).
- **BAAs.** Business associate agreements with every host that touches PHI.
- **Legal [COUNSEL].**
  - State recording-consent rules for the opening notice.
  - AI-disclosure wording.
  - FDA scope.
- **Reviews and gates.**
  - Clinician review of every caller-facing script.
  - Native-speaker review.
  - The open gates in the README: happy-path booking ≥ 90 %; for emergencies, zero misses on at least 60 cases per
    language.
- **Operations.**
  - Monitoring, alerting and a human on-call path for transfers.
  - Bounding the agent's in-memory per-call state.
