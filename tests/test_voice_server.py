import base64
import json

import numpy as np
import pytest

pytest.importorskip("pipecat")
from fastapi.testclient import TestClient  # noqa: E402
from loguru import logger  # noqa: E402
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402

from puente_voice.audit import AuditLog  # noqa: E402
from puente_voice.flow import CallFlow, State  # noqa: E402
from puente_voice.render import Renderer  # noqa: E402
from puente_voice.sim import demo_data  # noqa: E402
from puente_voice.voice.server import configure_logging, create_app  # noqa: E402

TONE = (np.sin(np.arange(24000) / 24000 * 2 * np.pi * 440) * 8000).astype(np.int16).tobytes()


class PassThrough(FrameProcessor):  # stands in for Whisper: no model, no transcripts
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def app(calls=None):
    return create_app(lambda cid: CallFlow(cid, demo_data.CLINIC, demo_data.scheduler(), AuditLog(),
                                           now=lambda: demo_data.NOW),
                      whisper_model=None, kokoro=None, prerendered={Renderer(demo_data.CLINIC).disclosure(): TONE},
                      prerendered_rate=24000, stt_factory=PassThrough, calls=calls)


def start(ws, sid):
    ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
    ws.send_text(json.dumps({"event": "start", "streamSid": f"MZ{sid}", "start": {
        "streamSid": f"MZ{sid}", "callSid": f"CA{sid}", "accountSid": "AC0", "tracks": ["inbound"],
        "customParameters": {}, "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))


def first_media(ws):
    msg = json.loads(ws.receive_text())
    assert msg["event"] == "media"
    return base64.b64decode(msg["media"]["payload"])


def test_twilio_caller_hears_the_disclosure_first():
    calls = {}
    with TestClient(app(calls)).websocket_connect("/twilio") as ws:
        start(ws, 1)
        assert any(b != 0xFF for b in first_media(ws))  # 0xFF is μ-law silence
        assert "CA1" in calls


def test_server_keeps_no_calls_unless_asked():
    a = app()
    with TestClient(a).websocket_connect("/twilio") as ws:
        start(ws, 1)
        first_media(ws)
    assert not hasattr(a.state, "calls")


def test_two_simultaneous_calls_get_separate_flows():
    calls = {}
    client = TestClient(app(calls))
    with client.websocket_connect("/twilio") as w1, client.websocket_connect("/twilio") as w2:
        start(w1, 1)
        start(w2, 2)
        first_media(w1)
        first_media(w2)
        assert calls["CA1"].flow is not calls["CA2"].flow


def test_caller_hangup_ends_the_call_without_booking():
    calls = {}
    with TestClient(app(calls)).websocket_connect("/twilio") as ws:
        start(ws, 1)
        first_media(ws)
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    flow = calls["CA1"].flow
    assert flow.booking is None and flow.state == State.LANGUAGE


def test_configure_logging_drops_debug_text(capsys):
    configure_logging()
    logger.debug("Transcription: [María Pérez 15 de marzo de 1980]")
    logger.info("call started")
    err = capsys.readouterr().err
    assert "María" not in err and "call started" in err


def test_vad_hears_a_one_syllable_answer():
    """A 0.39 s synthetic "sí" through the phone codec: Pipecat's default start_secs=0.2 never fires on it."""
    import asyncio
    from pathlib import Path

    from pipecat.audio.utils import create_stream_resampler, ulaw_to_pcm
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADState

    from puente_voice.flow import State
    from puente_voice.voice.flow_processor import vad_params

    async def states():
        ulaw = (Path(__file__).parent / "fixtures" / "si_es_8k.ulaw").read_bytes()
        pcm = await ulaw_to_pcm(ulaw, 8000, 8000, create_stream_resampler())
        vad = SileroVADAnalyzer(sample_rate=8000, params=vad_params(State.READBACK))
        vad.set_sample_rate(8000)
        audio = b"\x00\x00" * 8000 + pcm + b"\x00\x00" * 16000
        return [await vad.analyze_audio(audio[i:i + 320]) for i in range(0, len(audio) - 320, 320)]

    assert VADState.SPEAKING in asyncio.run(states())
