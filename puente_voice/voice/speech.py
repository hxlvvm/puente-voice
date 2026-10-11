"""Speech services: es/en Whisper, and Kokoro with fixed prompts rendered once per process."""
import asyncio

import numpy as np
import soxr
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import TranscriptionFrame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from puente_voice.render import TEMPLATES
from puente_voice.text import fold

KOKORO_LANG = {"es": "es", "en": "en-us"}
# What Whisper writes for line noise and coughs (caption text from its training data), never a caller's answer
HALLUCINATIONS = {"gracias", "gracias por ver el video", "suscribete", "thank you", "thanks for watching",
                  "thank you for watching", "subtitulos realizados por la comunidad de amaraorg"}
VOICES = {"es": "ef_dora", "en": "af_heart"}


def kokoro_providers(available):
    """The GPU when this onnxruntime build has one (CPU synthesis is the slowest step of a turn), then the CPU."""
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def load_kokoro(model_dir, threads):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1  # the default is every core on the node
    session = ort.InferenceSession(f"{model_dir}/kokoro-v1.0.onnx", o,
                                   providers=kokoro_providers(ort.get_available_providers()))
    logger.info(f"Kokoro on {session.get_providers()[0]}")
    return Kokoro.from_session(session, f"{model_dir}/voices-v1.0.bin")


def synth(kokoro, text, lang, voice=None):
    """16-bit PCM bytes and sample rate."""
    wav, sr = kokoro.create(text, voice=voice or VOICES[lang], lang=KOKORO_LANG[lang])
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes(), sr


def prerender(kokoro, renderer):
    """Audio for every prompt that never changes within a clinic, keyed by its exact text."""
    texts = [(t[lang], lang) for t in TEMPLATES.values() for lang in ("es", "en") if "{" not in t[lang]]
    texts += [(renderer.text("ask_visit", lang, options=renderer.visit_options(lang)), lang) for lang in ("es", "en")]
    cache, rate = {}, None
    for text, lang in texts:
        cache[text], rate = synth(kokoro, text, lang)
    parts = [synth(kokoro, text, lang) for text, lang in renderer.disclosure_parts()]
    cache[renderer.disclosure()] = b"".join(pcm for pcm, _ in parts)
    return cache, rate


class PuenteTTS(KokoroTTSService):
    """Kokoro TTS over one shared session; prompts found in `prerendered` play without synthesis."""

    def __init__(self, kokoro, prerendered, prerendered_rate, **kw):
        # shortcut: skips KokoroTTSService.__init__, which loads a fresh unthrottled model per call; recheck on upgrade
        TTSService.__init__(self, push_start_frame=True, push_stop_frames=True, settings=self.Settings(
            model=None, voice=VOICES["es"], language=Language.ES, speed=1.0), **kw)
        self._kokoro, self._pre, self._pre_rate = kokoro, prerendered, prerendered_rate
        self._resampler = create_stream_resampler()

    async def run_tts(self, text, context_id):
        pcm = self._pre.get(text)
        if pcm is None:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return
        audio = await self._resampler.resample(pcm, self._pre_rate, self.sample_rate)
        yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)


def pick_language(probs, preferred=None):
    """The call's language once set (the caller switches by asking); before that, Whisper's guess limited to es/en.

    Detection is unreliable on one-word answers: a phone-line "sí" scores 100 % English.
    """
    if preferred in ("es", "en"):
        return preferred
    p = dict(probs)
    return "es" if p.get("es", 0.0) >= p.get("en", 0.0) else "en"


def transcribe(model, x, lang=None):
    """One 16 kHz segment -> (language, text), with every filter the live pipeline applies.

    `lang` is the call's language once set; None detects es/en. Non-speech, low-confidence segments and Whisper's
    stock captions come back as "". The evaluation scripts call this too, so they measure what callers get.
    """
    lang = pick_language(model.detect_language(x)[2], lang)
    segments, _ = model.transcribe(x, language=lang, vad_filter=True)  # its Silero drops non-speech
    text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.4).strip()
    if "".join(c for c in fold(text) if c.isalpha() or c == " ").strip() in HALLUCINATIONS:
        text = ""
    return lang, text


class BilingualWhisper(WhisperSTTService):
    """faster-whisper over one shared model, es/en only. Never logs text: transcripts are PHI."""

    def __init__(self, model, **kw):
        self._shared = model  # set before super().__init__, which calls _load()
        super().__init__(settings=WhisperSTTService.Settings(model="shared", language=None), **kw)

    def _load(self):
        self._model = self._shared

    async def run_stt(self, audio):
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if self.sample_rate and self.sample_rate != 16000:  # faster-whisper assumes 16 kHz; phone audio is 8 kHz
            x = soxr.resample(x, self.sample_rate, 16000)
        lang, text = await asyncio.to_thread(transcribe, self._model, x, self._settings.language)
        logger.debug(f"{self}: {len(x) / 16000:.2f}s segment -> {lang}, {len(text)} chars")  # no text: PHI
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language(lang))
