"""Run the phone agent for a demo: the WebSocket server plus the Asterisk AudioSocket bridge.

NOT FOR PRODUCTION. Synthetic demo clinic and patients, a fixed demo clock (slots on 13-15 Oct 2026), no TLS, no
encryption at rest, scripts without clinician review. Settings come from the environment:

  PUENTE_MODELS            directory holding kokoro/ (scripts/fetch_models.sh)     required
  PUENTE_DEVICE            cpu or cuda                             default cpu
  PUENTE_WHISPER           faster-whisper model name or path       default: turbo on cuda, small on cpu
  PUENTE_SAFETY_MODEL      safety classifier directory              default: none, rules only
  PUENTE_AUDIT             audit log file (IDs only, no speech)     default: none
  PUENTE_HTTP_PORT         8080        PUENTE_AUDIOSOCKET_PORT 9092

usage: python -m puente_voice.serve
"""
import asyncio
import os

import uvicorn
from faster_whisper import WhisperModel

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow
from puente_voice.render import Renderer
from puente_voice.sim import demo_data
from puente_voice.telephony.audiosocket import bridge
from puente_voice.voice.server import configure_logging, create_app
from puente_voice.voice.speech import load_kokoro, prerender


async def main():
    env = os.environ.get
    configure_logging("INFO")  # DEBUG would log what callers say
    kokoro = load_kokoro(f"{env('PUENTE_MODELS')}/kokoro", len(os.sched_getaffinity(0)))
    pre, rate = prerender(kokoro, Renderer(demo_data.CLINIC))
    device = env("PUENTE_DEVICE", "cpu")
    # turbo on 4 CPU cores answered after the 8 s silence timer (job 7637774); small: accuracy unmeasured
    default = "deepdml/faster-whisper-large-v3-turbo-ct2" if device == "cuda" else "Systran/faster-whisper-small"
    whisper = WhisperModel(env("PUENTE_WHISPER", default), device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    safety = None
    if env("PUENTE_SAFETY_MODEL"):
        from puente_voice.safety_model import SafetyModel
        safety = SafetyModel(env("PUENTE_SAFETY_MODEL"))
    sched = demo_data.scheduler()  # one synthetic calendar shared by every demo call
    app = create_app(lambda cid: CallFlow(cid, demo_data.CLINIC, sched, AuditLog(env("PUENTE_AUDIT")),
                                          now=lambda: demo_data.NOW, safety_model=safety),
                     whisper, kokoro, pre, rate, outcomes={})
    http, audio = int(env("PUENTE_HTTP_PORT", "8080")), int(env("PUENTE_AUDIOSOCKET_PORT", "9092"))
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=http, log_level="warning"))
    sockets = await asyncio.start_server(lambda r, w: bridge(r, w, f"ws://127.0.0.1:{http}/twilio"), "0.0.0.0", audio)
    async with sockets:
        await asyncio.gather(server.serve(), sockets.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())
