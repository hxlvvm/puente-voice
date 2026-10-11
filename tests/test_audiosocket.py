"""The AudioSocket bridge against the real server, with a fake Asterisk on one side and a scripted STT on the other."""
import asyncio
import socket
import uuid

import httpx
import numpy as np
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow
from puente_voice.sim import demo_data
from puente_voice.telephony.audiosocket import AUDIO, DTMF, FRAME, HANGUP, UUID, bridge, frame, read_frame
from puente_voice.voice.server import create_app

TONE = (np.sin(np.arange(24000) / 24000 * 2 * np.pi * 440) * 8000).astype(np.int16).tobytes()  # 1 s at 24 kHz
SILENCE = b"\x00" * FRAME


class AnyPrompt(dict):  # every prompt is pre-rendered as a tone: no Kokoro needed
    def __init__(self, tone=TONE):
        super().__init__()
        self.tone = tone

    def get(self, key, default=None):
        return self.tone


def scripted_stt(text, after_frames):
    class Scripted(FrameProcessor):  # stands in for Whisper: says `text` once, after_frames into the call
        n = 0

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame) and text:
                Scripted.n += 1
                if Scripted.n == after_frames:
                    await self.push_frame(TranscriptionFrame(text, "caller", "2026-10-12T00:00:00Z"), direction)
    return Scripted


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def call(text=None, hang_up_after_s=None, tone=TONE, arrivals=None):
    """One call through the bridge; returns (agent audio heard, agent hung up, outcome, outcome read again).

    `arrivals`, if given, collects the arrival time of every audio frame the caller receives."""
    import uvicorn
    outcomes = {}
    app = create_app(lambda cid: CallFlow(cid, demo_data.CLINIC, demo_data.scheduler(), AuditLog(),
                                          now=lambda: demo_data.NOW),
                     whisper_model=None, kokoro=None, prerendered=AnyPrompt(tone), prerendered_rate=24000,
                     idle_s=30, stt_factory=scripted_stt(text, 150), outcomes=outcomes)
    http, tcp = free_port(), free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=http, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    bridges = await asyncio.start_server(lambda r, w: bridge(r, w, f"ws://127.0.0.1:{http}/twilio"), "127.0.0.1", tcp)
    reader, writer = await asyncio.open_connection("127.0.0.1", tcp)
    sid = uuid.uuid4()
    writer.write(frame(UUID, sid.bytes))
    heard, agent_hung_up = bytearray(), False

    async def talk():  # the line carries 20 ms frames in real time, like Asterisk
        for i in range(int((hang_up_after_s or 15) / 0.02)):
            writer.write(frame(DTMF, b"5") if i == 60 else frame(AUDIO, SILENCE))  # a key press is not a hang-up
            await writer.drain()
            await asyncio.sleep(0.02)
        if hang_up_after_s:
            writer.write(frame(HANGUP))
            await writer.drain()

    talking = asyncio.create_task(talk())
    try:
        while True:
            kind, payload = await asyncio.wait_for(read_frame(reader), 20)
            if kind == AUDIO:
                assert len(payload) <= FRAME
                heard += payload
                if arrivals is not None:
                    arrivals.append(asyncio.get_running_loop().time())
            elif kind == HANGUP:
                agent_hung_up = True
    except asyncio.IncompleteReadError:
        pass  # the bridge closed the connection
    talking.cancel()
    writer.close()
    async with httpx.AsyncClient() as c:
        outcome = (await c.get(f"http://127.0.0.1:{http}/outcome/{sid}")).text
        again = (await c.get(f"http://127.0.0.1:{http}/outcome/{sid}")).text
    bridges.close()
    server.should_exit = True
    await serving
    return bytes(heard), agent_hung_up, outcome, again


def test_caller_hears_the_disclosure_and_a_staff_request_ends_in_transfer():
    heard, agent_hung_up, outcome, again = asyncio.run(call("quiero hablar con una persona"))
    pcm = np.frombuffer(heard, np.int16)
    assert np.abs(pcm).max() > 1000                  # the disclosure tone reached the caller as audio
    assert agent_hung_up and outcome == "transferred"  # the dialplan goes on to ring staff
    assert again == "unknown"                          # read once, then forgotten


def test_caller_hang_up_ends_the_call_without_transfer():
    heard, agent_hung_up, outcome, _ = asyncio.run(call(hang_up_after_s=2))
    assert len(heard) > 0 and not agent_hung_up and outcome == "language"


def test_frames_round_trip():
    async def go():
        r = asyncio.StreamReader()
        r.feed_data(frame(AUDIO, b"\x01\x02") + frame(HANGUP))
        r.feed_eof()
        return await read_frame(r), await read_frame(r)
    assert asyncio.run(go()) == ((AUDIO, b"\x01\x02"), (HANGUP, b""))


def test_agent_audio_arrives_in_real_time():
    """Asterisk plays frames as they arrive: audio sent ahead of time would pile up in the caller's jitter buffer."""
    arrivals = []
    five_s = np.tile(TONE, 5)
    heard, *_ = asyncio.run(call(hang_up_after_s=9, tone=five_s, arrivals=arrivals))
    audio_s, wall_s = len(heard) / 16000, arrivals[-1] - arrivals[0]
    assert audio_s > 4.5 and 0.8 < audio_s / wall_s < 1.25, (audio_s, wall_s)
