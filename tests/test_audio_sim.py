import numpy as np
import pytest

pytest.importorskip("pipecat")
from puente_voice.sim.audio_sim import add_noise, rms, utterance_pcm, wer_counts  # noqa: E402


def test_wer_counts_words_after_folding():
    assert wer_counts("Me llamo María Pérez.", "me llamo maria perez") == (0, 4)
    assert wer_counts("quince de marzo", "15 de marzo") == (1, 3)
    assert wer_counts("sí", "") == (1, 1)


def test_add_noise_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    speech = (np.sin(np.arange(8000) / 8000 * 2 * np.pi * 300) * 3000).astype(np.int16).tobytes()
    noise_rms = rms(speech) / 10 ** (10 / 20)  # 10 dB
    noisy = add_noise(speech, noise_rms, rng)
    diff = np.frombuffer(noisy, np.int16).astype(np.float64) - np.frombuffer(speech, np.int16)
    assert abs(20 * np.log10(rms(speech) / np.sqrt((diff ** 2).mean())) - 10) < 0.5


class FakeKokoro:
    def create(self, text, voice, lang):
        return np.full(24000, 0.2, dtype=np.float32), 24000  # 1 s


def test_pause_marker_inserts_400_ms_of_silence():
    plain = utterance_pcm(FakeKokoro(), "15 de marzo de 1980", "es")
    paused = utterance_pcm(FakeKokoro(), "15 de marzo | de 1980", "es")
    assert abs(len(plain) - 8000 * 2) <= 8 and len(paused) - 2 * len(plain) == 3200 * 2


def test_caller_stops_sending_once_the_server_hangs_up():
    import asyncio

    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close

    from puente_voice.sim.audio_sim import Caller

    class HungUp:
        sent = 0

        async def send(self, data):
            self.sent += 1
            raise ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)

    ws = HungUp()
    c = Caller(ws, [b"\xff" * 160])
    c.sid = "MZ1"
    asyncio.run(c.wait(lambda: False, timeout=1))
    assert c.closed and ws.sent == 1


def test_final_prompt_is_measured_from_its_own_first_frame():
    from puente_voice.sim.audio_sim import last_speech_span
    # 40 ms frames: an interrupted prompt (0-1 s), a pause, then the final prompt (3.0-6.6 s)
    voiced = [i * 0.04 for i in range(26)] + [3.0 + i * 0.04 for i in range(91)]
    assert abs(last_speech_span(voiced) - 3.64) < 0.05
    assert last_speech_span([]) == 0.0


def test_merge_reports_over_all_shards(tmp_path, capsys):
    import json

    from puente_voice.sim.audio_sim import main

    def call(name, passed, lat):
        return {"name": name, "lang": "es", "passed": passed, "wrong_booking": False, "latencies": [lat],
                "wer_edits": 1, "wer_words": 10}
    (tmp_path / "a.jsonl").write_text(json.dumps(call("a", True, 0.5)) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(call("b", False, 1.5)) + "\n")
    assert main(["--merge", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]) == 1
    out = capsys.readouterr().out
    assert "1/2 scenarios passed" in out and "WER es 10.0% over 20 words" in out


def test_wer_reference_counts_only_turns_that_were_spoken():
    from puente_voice.sim.audio_sim import wer_reference
    assert wer_reference(["hola", "", "15 de marzo | de 1980", "sí", "la primera"], 3) == "hola 15 de marzo   de 1980"
