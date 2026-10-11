"""Phone-call server: a Twilio Media Streams WebSocket in, one Pipecat pipeline per call."""
import asyncio
import sys

from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from puente_voice.flow import State
from puente_voice.voice.flow_processor import FlowProcessor, vad_params
from puente_voice.voice.speech import BilingualWhisper, PuenteTTS


def configure_logging(level="INFO"):
    """Pipecat logs transcripts and TTS text at DEBUG; keep it off."""
    logger.remove()
    logger.add(sys.stderr, level=level)


def create_app(make_flow, whisper_model, kokoro, prerendered, prerendered_rate, idle_s=8.0, stt_factory=None,
               calls=None, outcomes=None):
    """`calls`, if given, collects each call's FlowProcessor by call id: simulator only, it holds transcripts.

    `outcomes`, if given, records each finished call's final state ("transferred", "done", ...) for GET /outcome/<id>,
    which the Asterisk dialplan reads once to decide whether to ring staff.
    """
    app = FastAPI()

    @app.get("/outcome/{call_id}", response_class=PlainTextResponse)
    async def outcome(call_id: str):
        for _ in range(40):  # Asterisk asks the moment the socket closes, a few ms before the worker has returned
            if call_id in (outcomes or {}):
                return outcomes.pop(call_id)
            await asyncio.sleep(0.05)
        return "unknown"

    @app.websocket("/twilio")
    async def twilio(websocket: WebSocket):
        await websocket.accept()
        _, call = await parse_telephony_websocket(websocket)
        serializer = TwilioFrameSerializer(  # no Twilio credentials: hang-up and transfer are milestone 4
            stream_sid=call["stream_id"], call_sid=call["call_id"],
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False))
        transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer))
        flow = FlowProcessor(make_flow(call["call_id"]), idle_s=idle_s)
        if calls is not None:
            calls[call["call_id"]] = flow
        stt = stt_factory() if stt_factory else BilingualWhisper(whisper_model)
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params(State.LANGUAGE)))
        worker = PipelineWorker(
            Pipeline([transport.input(), vad, stt, flow, PuenteTTS(kokoro, prerendered, prerendered_rate),
                      transport.output()]),
            params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000))

        @transport.event_handler("on_client_connected")
        async def _(t, c):
            await worker.queue_frames(flow.opening())

        @transport.event_handler("on_client_disconnected")
        async def _(t, c):
            await worker.cancel()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()
        if outcomes is not None:
            # shortcut: a call whose outcome is never read stays here (a few bytes); bound it before real traffic
            outcomes[call["call_id"]] = flow.flow.state.value

    return app
