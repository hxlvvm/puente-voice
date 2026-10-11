import asyncio

import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndWorkerFrame,  # noqa: E402
                                   TranscriptionFrame, TTSSpeakFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

from puente_voice.audit import AuditLog  # noqa: E402
from puente_voice.flow import CallFlow, State  # noqa: E402
from puente_voice.sim import demo_data  # noqa: E402
from puente_voice.voice.flow_processor import FlowProcessor  # noqa: E402

TO_READBACK = ["Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "consulta general",
               "la primera"]


def proc(idle_s=8.0, turns=()):
    f = CallFlow("call-p", demo_data.CLINIC, demo_data.scheduler(), AuditLog(), now=lambda: demo_data.NOW)
    f.start()
    for t in turns:
        f.handle(t)
    return FlowProcessor(f, idle_s=idle_s)


def tx(text):
    return TranscriptionFrame(text, "caller", "2026-10-09T00:00:00Z")


def spoken(frames):
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


def interrupted(down, up):
    return any("Interruption" in type(f).__name__ for f in list(down) + list(up))


def test_opening_plays_the_disclosure_with_the_spanish_voice():
    p = proc()
    frames = FlowProcessor(CallFlow("c", demo_data.CLINIC, demo_data.scheduler(), AuditLog(),
                                    now=lambda: demo_data.NOW)).opening()
    assert [f.text for f in frames if isinstance(f, TTSSpeakFrame)] == [p.flow.r.disclosure()]


def test_speech_before_the_disclosure_ends_is_ignored():
    p = proc()
    down, _ = asyncio.run(run_test(p, frames_to_send=[tx("Hola, quiero una cita")]))
    assert spoken(down) == [] and p.flow.state == State.LANGUAGE and p.heard == []


def test_after_the_disclosure_a_transcript_drives_the_flow():
    p = proc()
    down, _ = asyncio.run(run_test(p, frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(),
                                                      tx("Hola, quiero una cita")]))
    assert spoken(down) == ["Por favor, dígame su nombre completo."] and p.heard == ["Hola, quiero una cita"]


def test_silent_caller_gets_an_empty_turn_after_idle():
    p = proc(idle_s=0.05)
    down, _ = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), SleepFrame(sleep=0.2)]))
    assert spoken(down) and "repetir" in spoken(down)[0] and p.flow.fails == 1


def test_barge_in_interrupts_the_prompt():
    p = proc()
    down, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), BotStartedSpeakingFrame(),
                                                       VADUserStartedSpeakingFrame()]))
    assert interrupted(down, up)


def test_caller_cannot_talk_over_the_emergency_script():
    p = proc()
    # system frames overtake queued data frames, so pause until the transcript has been handled
    down, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), tx("me duele mucho el pecho"),
                                                       SleepFrame(sleep=0.05), BotStartedSpeakingFrame(),
                                                       VADUserStartedSpeakingFrame()]))
    assert p.flow.transfer_reason == "emergency:chest_pain" and not interrupted(down, up)


def test_no_idle_turn_while_the_caller_is_still_talking():
    p = proc(idle_s=0.05)
    asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), BotStartedSpeakingFrame(),
                                            VADUserStartedSpeakingFrame(), BotStoppedSpeakingFrame(),
                                            SleepFrame(sleep=0.2)]))
    assert p.flow.fails == 0


def test_noise_with_no_words_restarts_the_idle_timer():
    p = proc(idle_s=0.05)
    asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), VADUserStartedSpeakingFrame(),
                                            VADUserStoppedSpeakingFrame(), SleepFrame(sleep=0.2)]))
    assert p.flow.fails == 1  # the line does not go dead after a cough


def test_yes_over_the_readback_question_books_nothing():
    p = proc(turns=TO_READBACK)
    assert p.flow.state == State.READBACK
    down, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), BotStartedSpeakingFrame(),
                                                       VADUserStartedSpeakingFrame(), tx("sí")]))
    assert p.flow.booking is None and not interrupted(down, up) and p.flow.fails == 0


def test_yes_after_the_readback_question_books():
    p = proc(turns=TO_READBACK)
    down, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), VADUserStartedSpeakingFrame(),
                                                       tx("sí"), SleepFrame(sleep=0.05), BotStartedSpeakingFrame(),
                                                       BotStoppedSpeakingFrame()]))  # the goodbye plays, then hang-up
    assert p.flow.booking and p.flow.state == State.DONE and any(isinstance(f, EndWorkerFrame) for f in up)


def test_transfer_hangs_up_after_speaking():
    p = proc()
    down, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(),
                                                       tx("Quiero hablar con una persona")]))
    assert spoken(down) and any(isinstance(f, EndWorkerFrame) for f in up) and p.flow.state == State.TRANSFERRED


def test_the_call_language_is_passed_to_speech_recognition_once():
    from pipecat.frames.frames import STTUpdateSettingsFrame
    p = proc()
    _, up = asyncio.run(run_test(p, frames_to_send=[BotStoppedSpeakingFrame(), tx("Hello, I need an appointment"),
                                                    SleepFrame(sleep=0.05), tx("My name is John Smith")]))
    assert [f.delta.language for f in up if isinstance(f, STTUpdateSettingsFrame)] == ["en"]


def test_vad_waits_longer_while_the_caller_gives_a_name_or_a_date():
    from pipecat.frames.frames import VADParamsUpdateFrame
    p = proc()
    _, up = asyncio.run(run_test(p, frames_to_send=[
        BotStoppedSpeakingFrame(), tx("Hola, quiero una cita"), SleepFrame(sleep=0.05), tx("Me llamo María Pérez"),
        SleepFrame(sleep=0.05), tx("15 de marzo de 1980")]))
    assert p.flow.state == State.DOB_CONFIRM
    params = [(f.params.start_secs, f.params.stop_secs) for f in up if isinstance(f, VADParamsUpdateFrame)]
    assert params == [(0.2, 1.0), (0.1, 0.5)]  # quick start only where a one-word yes/no is expected


# Final review findings (probes P1-P8): BX/BS bot stopped/started, US/UX caller started/stopped
BX, BS = BotStoppedSpeakingFrame, BotStartedSpeakingFrame
US, UX = VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame


def test_p1_yes_before_the_readback_has_played_books_nothing():
    p = proc(turns=TO_READBACK[:-1])
    asyncio.run(run_test(p, frames_to_send=[BX(), tx("la tercera"), SleepFrame(sleep=0.05), US(), tx("sí")]))
    assert p.flow.state == State.READBACK and p.flow.booking is None


def test_p2_a_dropped_answer_still_gets_a_reask():
    p = proc(idle_s=0.05, turns=TO_READBACK)
    down, _ = asyncio.run(run_test(p, frames_to_send=[BX(), BS(), US(), BX(), UX(), tx("sí"), SleepFrame(sleep=0.5)]))
    assert p.flow.booking is None and spoken(down)


def test_p3_a_later_utterance_does_not_clear_a_talk_over():
    p = proc(turns=TO_READBACK)
    asyncio.run(run_test(p, frames_to_send=[BX(), BS(), US(), BX(), UX(), US(), tx("sí"), UX(), tx("no, espere")]))
    assert p.flow.booking is None


def test_p4_language_is_not_pinned_before_the_caller_chooses():
    from pipecat.frames.frames import STTUpdateSettingsFrame
    p = proc()
    _, up = asyncio.run(run_test(p, frames_to_send=[BX(), tx("Hmm okay"), SleepFrame(sleep=0.05)]))
    assert p.flow.state == State.LANGUAGE and not [f for f in up if isinstance(f, STTUpdateSettingsFrame)]


def test_p5_speech_begun_during_the_disclosure_is_never_acted_on():
    p = proc()
    asyncio.run(run_test(p, frames_to_send=[BS(), US(), BX(), UX(), tx("Hola quiero una cita")]))
    assert p.flow.state == State.LANGUAGE and p.heard == []


def test_recognition_is_muted_until_the_disclosure_has_played():
    from pipecat.frames.frames import STTMuteFrame
    p = proc()
    first = FlowProcessor(CallFlow("c", demo_data.CLINIC, demo_data.scheduler(), AuditLog(),
                                   now=lambda: demo_data.NOW)).opening()[0]
    _, up = asyncio.run(run_test(p, frames_to_send=[BX()]))
    assert isinstance(first, STTMuteFrame) and first.mute
    assert [f.mute for f in up if isinstance(f, STTMuteFrame)] == [False]


def test_p6_emergency_over_the_readback_question_still_transfers():
    p = proc(turns=TO_READBACK)
    asyncio.run(run_test(p, frames_to_send=[BX(), BS(), US(), tx("me duele mucho el pecho")]))
    assert p.flow.transfer_reason == "emergency:chest_pain"


def test_p7_persona_over_the_dob_question_still_transfers():
    p = proc(turns=TO_READBACK[:3])
    assert p.flow.state == State.DOB_CONFIRM
    asyncio.run(run_test(p, frames_to_send=[BX(), BS(), US(), tx("quiero hablar con una persona")]))
    assert p.flow.transfer_reason == "caller_request"


def test_p8_emergency_during_the_goodbye_still_transfers():
    p = proc(turns=TO_READBACK)
    asyncio.run(run_test(p, frames_to_send=[BX(), US(), tx("sí"), SleepFrame(sleep=0.05), BS(), US(), UX(),
                                            tx("espere, me duele mucho el pecho")]))
    assert p.flow.booking and p.flow.transfer_reason == "emergency:chest_pain"
