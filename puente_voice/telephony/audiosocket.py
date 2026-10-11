"""Asterisk AudioSocket in, the server's Twilio Media Streams endpoint out: one bridge connection per call.

Asterisk's dialplan app `AudioSocket(<uuid>,<host>:<port>)` speaks frames of 1 kind byte, a 2-byte big-endian length
and the payload: 0x01 the call's UUID (16 bytes), 0x10 audio (signed linear 16-bit, 8 kHz, mono, little-endian),
0x00 hang-up, 0xff error; other kinds (DTMF) are ignored. The bridge turns them into the Twilio messages the server already handles, so the call
flow, speech and timing code is the same one the simulator measures.
"""
import asyncio
import base64
import json
import uuid

import websockets
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm

HANGUP, UUID, DTMF, AUDIO, ERROR = 0x00, 0x01, 0x03, 0x10, 0xff
FRAME = 320  # 20 ms of 8 kHz 16-bit audio


def frame(kind, payload=b""):
    return bytes([kind]) + len(payload).to_bytes(2, "big") + payload


async def read_frame(reader):
    head = await reader.readexactly(3)
    return head[0], await reader.readexactly(int.from_bytes(head[1:], "big"))


async def bridge(reader, writer, ws_url):
    """Serve one AudioSocket connection until either side ends the call."""
    try:
        kind, payload = await read_frame(reader)
        if kind != UUID:
            return
        sid = str(uuid.UUID(bytes=payload))
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            await ws.send(json.dumps({"event": "start", "streamSid": sid, "start": {
                "streamSid": sid, "callSid": sid, "accountSid": "asterisk", "tracks": ["inbound"],
                "customParameters": {},
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))

            async def caller_to_agent():
                enc = create_stream_resampler()
                while True:
                    kind, payload = await read_frame(reader)
                    if kind in (HANGUP, ERROR):  # the caller is gone
                        break
                    if kind != AUDIO:  # e.g. DTMF from newer Asterisk versions: the flow takes speech only
                        continue
                    u = await pcm_to_ulaw(payload, 8000, 8000, enc)
                    await ws.send(json.dumps({"event": "media", "streamSid": sid,
                                              "media": {"payload": base64.b64encode(u).decode()}}))
                await ws.send(json.dumps({"event": "stop", "streamSid": sid}))

            async def agent_to_caller():
                dec = create_stream_resampler()
                # the server paces its audio in real time, so nothing queues here and a "clear" has nothing to flush
                async for message in ws:
                    m = json.loads(message)
                    if m.get("event") == "media":
                        pcm = await ulaw_to_pcm(base64.b64decode(m["media"]["payload"]), 8000, 8000, dec)
                        writer.write(b"".join(frame(AUDIO, pcm[i:i + FRAME]) for i in range(0, len(pcm), FRAME)))
                        await writer.drain()
                writer.write(frame(HANGUP))  # the agent ended the call: the dialplan continues
                await writer.drain()

            tasks = [asyncio.create_task(caller_to_agent()), asyncio.create_task(agent_to_caller())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                t.result()  # a dropped connection ends the call quietly (below); anything else is a bug and raises
    except (asyncio.IncompleteReadError, ConnectionError, websockets.ConnectionClosed):
        pass  # either side dropped the connection: the call is over
    finally:
        writer.close()
