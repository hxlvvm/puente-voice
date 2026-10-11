"""Pipecat processor that runs CallFlow: transcripts in, speech and hang-up frames out."""
import asyncio

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame, EndFrame,
                                   EndWorkerFrame, STTMuteFrame, STTUpdateSettingsFrame, TranscriptionFrame,
                                   TTSSpeakFrame, TTSUpdateSettingsFrame, VADParamsUpdateFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language

from puente_voice.flow import End, Say, State, Transfer
from puente_voice.voice.speech import VOICES

LANGS = {"es": Language.ES, "en": Language.EN, "bi": Language.ES}  # "bi" is pre-rendered, voices per part
CONFIRM = (State.DOB_CONFIRM, State.READBACK)  # a yes here only counts once the question has been heard in full
YES_NO = CONFIRM + (State.NEW_PATIENT,)
SLOW = (State.NAME, State.DOB, State.PHONE)


def vad_params(state):
    """Turn detection for the answer the flow is waiting for.

    start 0.1 s only where a one-syllable "sí" (0.39 s) is expected: Pipecat's 0.2 s never fires on it, but elsewhere
    0.1 s lets coughs barge in. stop 1.0 s while callers give a name, date or phone number, which they pause inside
    ("15 de marzo… de 1980"); 0.5 s otherwise.
    """
    return VADParams(start_secs=0.1 if state in YES_NO else 0.2, stop_secs=1.0 if state in SLOW else 0.5)


class FlowProcessor(FrameProcessor):
    """Deaf until the disclosure has played; a silent caller becomes an empty turn after idle_s."""

    def __init__(self, flow, idle_s=8.0):
        super().__init__()
        self.flow, self.idle_s = flow, idle_s
        self.listening = self.bot_speaking = self.user_speaking = self.ended = self.talked_over = False
        self.heard, self.said = [], []  # for the simulator; memory only, never logged
        self._idle, self._stt_lang, self._vad = None, None, vad_params(flow.state)
        self._unheard = self._early = self._end_pending = False

    def opening(self):
        # nothing is transcribed before the disclosure has played (spec, call flow step 1)
        return [STTMuteFrame(mute=True)] + self._frames(self.flow.start())

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_idle()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
            await self._stop_idle()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if not self.listening:  # the first stop is the end of the disclosure
                await self.push_frame(STTMuteFrame(mute=False), FrameDirection.UPSTREAM)
            self.bot_speaking, self.listening, self._unheard = False, True, False
            if self._end_pending:  # the goodbye has played in full
                self._end_pending = False
                await self.push_frame(EndWorkerFrame(), FrameDirection.UPSTREAM)
            elif not self.user_speaking:
                await self._start_idle()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.user_speaking = False
            if self.listening and not self.bot_speaking:
                await self._start_idle()  # cancelled by the transcript; fires if the sound had no words
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.user_speaking = True
            if not self.listening:
                self._early = True  # began during the disclosure: never acted on
            else:
                # sticky until a transcript consumes it: the next utterance's start can arrive before this transcript
                self.talked_over = self.talked_over or self.bot_speaking or self._unheard
                await self._stop_idle()
                if self.bot_speaking and not self.ended and self.flow.state not in CONFIRM:
                    await self.broadcast_interruption()  # barge-in; never over the goodbye or the emergency script
        if isinstance(frame, TranscriptionFrame):
            await self._on_transcript(frame.text)
            return
        await self.push_frame(frame, direction)

    async def _on_transcript(self, text):
        await self._stop_idle()
        early, over = self._early, self.talked_over
        self._early = self.talked_over = False
        if not self.listening or early:
            return
        if over and self.flow.state in CONFIRM and not (self.flow.safety.check(text) or self.flow.ip.persona(text)):
            await self._start_idle()  # a yes said over the question does not count; the caller still gets a re-ask
            return
        self.heard.append(text)
        await self._act(self.flow.handle(text))
        if self.flow.state != State.LANGUAGE and self.flow.lang != self._stt_lang:
            self._stt_lang = self.flow.lang  # recognition sticks to the call's language (speech.pick_language)
            await self.push_frame(STTUpdateSettingsFrame(delta=WhisperSTTService.Settings(
                language=LANGS[self.flow.lang])), FrameDirection.UPSTREAM)

    async def _start_idle(self):
        await self._stop_idle()
        if not self.ended:
            self._idle = self.create_task(self._idle_turn())

    async def _idle_turn(self):
        await asyncio.sleep(self.idle_s)
        self._idle, self.talked_over = None, False
        await self._act(self.flow.handle(""))

    async def _stop_idle(self):
        if self._idle:
            await self.cancel_task(self._idle)
            self._idle = None

    async def _act(self, actions):
        params = vad_params(self.flow.state)
        if params != self._vad:
            self._vad = params
            await self.push_frame(VADParamsUpdateFrame(params), FrameDirection.UPSTREAM)
        for f in self._frames(actions):
            up = isinstance(f, EndWorkerFrame)
            await self.push_frame(f, FrameDirection.UPSTREAM if up else FrameDirection.DOWNSTREAM)
        if self.flow.state in CONFIRM and any(isinstance(a, Say) for a in actions):
            self._unheard = True  # until it has played, nobody can have answered this question

    def _frames(self, actions):
        out = []
        for a in actions:
            if isinstance(a, Say):
                self.said.append(a.text)
                voice = VOICES.get(a.lang, VOICES["es"])
                out += [TTSUpdateSettingsFrame(delta=KokoroTTSService.Settings(voice=voice, language=LANGS[a.lang])),
                        TTSSpeakFrame(a.text)]
            elif isinstance(a, Transfer):
                self.ended, self._end_pending = True, False
                out.append(EndWorkerFrame())  # the worker ends after the queued speech has played
            elif isinstance(a, End):
                # hang up only after the goodbye has played, so an emergency said over it still reaches the flow
                self.ended = self._end_pending = True
        return out
