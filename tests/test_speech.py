import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402

from puente_voice.render import Renderer  # noqa: E402
from puente_voice.sim import demo_data  # noqa: E402
from puente_voice.voice.speech import BilingualWhisper, PuenteTTS, pick_language, prerender  # noqa: E402

R = Renderer(demo_data.CLINIC)
TONE = (np.sin(np.arange(24000) / 24000 * 2 * np.pi * 440) * 8000).astype(np.int16).tobytes()  # 1 s at 24 kHz


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def create(self, text, voice, lang):
        self.calls.append((text, voice, lang))
        return np.full(2400, 0.1, dtype=np.float32), 24000  # 0.1 s per call


def test_pick_language_is_es_or_en():
    assert pick_language([("pt", 0.6), ("es", 0.3), ("en", 0.1)]) == "es"
    assert pick_language([("en", 0.5), ("es", 0.4)]) == "en"
    assert pick_language([("it", 0.9)]) == "es"


def test_prerender_covers_fixed_prompts_and_speaks_each_disclosure_part_in_its_language():
    k = FakeKokoro()
    cache, rate = prerender(k, R)
    assert rate == 24000
    assert R.text("ask_name", "es") in cache and R.text("ask_name", "en") in cache
    assert R.text("ask_visit", "es", options=R.visit_options("es")) in cache
    assert all("{" not in text for text in cache)
    assert len(cache[R.disclosure()]) == len(R.disclosure_parts()) * 2400 * 2  # one fake tone per part, 16-bit samples
    parts = [(t, lang) for t, _, lang in k.calls if t in dict(R.disclosure_parts())]
    assert [lang for _, lang in parts] == ["es", "en-us"] * 3


def test_prerendered_prompt_plays_without_synthesis():
    tts = PuenteTTS(kokoro=None, prerendered={"Hola.": TONE}, prerendered_rate=24000)
    down, _ = asyncio.run(run_test(tts, frames_to_send=[TTSSpeakFrame("Hola.")]))
    assert sum(len(f.audio) for f in down if isinstance(f, TTSAudioRawFrame)) > 0


def test_english_voice_update_reaches_kokoro_as_en_us():
    from pipecat.frames.frames import TTSUpdateSettingsFrame
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.transcriptions.language import Language
    tts = PuenteTTS(kokoro=None, prerendered={"Hi.": TONE}, prerendered_rate=24000)
    asyncio.run(run_test(tts, frames_to_send=[TTSUpdateSettingsFrame(delta=KokoroTTSService.Settings(
        voice="af_heart", language=Language.EN)), TTSSpeakFrame("Hi.")]))
    assert tts._settings.language == "en-us" and tts._settings.voice == "af_heart"  # kokoro-onnx wants en-us


class FakeWhisper:
    def detect_language(self, x):
        return "en", 0.9, [("en", 0.9), ("es", 0.1)]

    def transcribe(self, x, language, **kw):
        assert language == "en"
        return [SimpleNamespace(text=" Hello there.", no_speech_prob=0.1),
                SimpleNamespace(text=" noise", no_speech_prob=0.9)], None


def test_bilingual_whisper_keeps_speech_segments_only():
    async def go():
        return [f async for f in BilingualWhisper(FakeWhisper()).run_stt(b"\x00\x00" * 1600)]
    (frame,) = asyncio.run(go())
    assert frame.text == "Hello there." and frame.language == "en"


def test_whisper_receives_16_khz_audio():
    """faster-whisper assumes 16 kHz; phone audio arrives at 8 kHz and must be resampled, not sped up."""
    class Recorder(FakeWhisper):
        def transcribe(self, x, language, **kw):
            self.samples = len(x)
            return super().transcribe(x, language, **kw)

    model = Recorder()
    stt = BilingualWhisper(model)
    stt._sample_rate = 8000

    async def go():
        return [f async for f in stt.run_stt(b"\x00\x00" * 8000)]  # one second at 8 kHz
    asyncio.run(go())
    assert abs(model.samples - 16000) <= 16


def test_pick_language_keeps_the_call_language_once_set():
    # Whisper rates a 0.4 s phone-line "sí" 100 % English ("See you."); forced to Spanish it reads "Sí." correctly, and
    # English sentences survive Spanish decoding, so a switch comes from the caller's request (switch_request), not sound
    assert pick_language([("en", 1.0), ("es", 0.0)], preferred="es") == "es"
    assert pick_language([("es", 0.99), ("en", 0.01)], preferred="en") == "en"
    assert pick_language([("en", 0.6), ("es", 0.3)]) == "en"  # before the call has a language: detect


class ShortSi:
    def detect_language(self, x):
        return "en", 0.6, [("en", 0.6), ("es", 0.3)]

    def transcribe(self, x, language, **kw):
        self.language = language
        return [SimpleNamespace(text=" Sí.", no_speech_prob=0.1)], None


def test_whisper_keeps_the_call_language_set_by_the_flow():
    from pipecat.frames.frames import STTUpdateSettingsFrame
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language

    model = ShortSi()
    stt = BilingualWhisper(model)
    asyncio.run(run_test(stt, frames_to_send=[STTUpdateSettingsFrame(
        delta=WhisperSTTService.Settings(language=Language.ES))]))

    async def go():
        return [f async for f in stt.run_stt(b"\x00\x00" * 1600)]
    (frame,) = asyncio.run(go())
    assert model.language == "es" and frame.language == "es"


def test_kokoro_runs_on_the_gpu_when_the_runtime_has_one():
    from puente_voice.voice.speech import kokoro_providers
    assert kokoro_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert kokoro_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]


def test_noise_is_filtered_and_stock_hallucinations_are_dropped():
    class Hallucinating(FakeWhisper):
        def transcribe(self, x, language, **kw):
            self.kw = kw
            return [SimpleNamespace(text=" Gracias por ver el video.", no_speech_prob=0.0)], None

    model = Hallucinating()

    async def go():
        return [f async for f in BilingualWhisper(model).run_stt(b"\x00\x00" * 1600)]
    assert asyncio.run(go()) == [] and model.kw.get("vad_filter") is True


def test_shared_transcribe_matches_the_live_filters():
    from puente_voice.voice.speech import transcribe

    class Stub:
        def __init__(self, text, probs):
            self.text, self.probs, self.forced = text, probs, None

        def detect_language(self, x):
            return self.probs[0][0], self.probs[0][1], self.probs

        def transcribe(self, x, language, **kw):
            self.forced, self.kw = language, kw
            return [SimpleNamespace(text=self.text, no_speech_prob=0.1)], None

    x = np.zeros(16000, dtype=np.float32)
    m = Stub(" ¡Gracias!", [("es", 0.9), ("en", 0.1)])
    assert transcribe(m, x, "es") == ("es", "") and m.kw.get("vad_filter") is True  # stock caption dropped
    m = Stub(" Hello there.", [("en", 0.8), ("es", 0.2)])
    assert transcribe(m, x) == ("en", "Hello there.") and m.forced == "en"  # no call language yet: detected
