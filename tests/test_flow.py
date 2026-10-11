from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow, End, Say, State, Transfer
from puente_voice.sim import demo_data


def new_flow(sched=None):
    sched = sched or demo_data.scheduler()
    audit = AuditLog()
    return CallFlow("call-1", demo_data.CLINIC, sched, audit, now=lambda: demo_data.NOW), sched, audit


def say(actions):
    return " ".join(a.text for a in actions if isinstance(a, Say))


def run(flow, *turns):
    out = []
    for t in turns:
        out = flow.handle(t)
    return out


def test_start_plays_disclosure_and_audits_it():
    f, _, audit = new_flow()
    acts = f.start()
    assert "asistente automático" in say(acts) and audit.events[0]["kind"] == "disclosure_played"


def test_happy_path_spanish_books_after_readback():
    f, sched, _ = new_flow()
    f.start()
    acts = run(f, "Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "consulta general",
               "la primera", "sí")
    assert f.state == State.DONE and isinstance(acts[-1], End)
    assert f.booking.slot_id == "s1" and f.booking.patient_id == "p1"
    assert "quedó reservada el martes 13 de octubre a las 10:00 de la mañana" in say(acts)


def test_readback_cannot_be_skipped():
    f, sched, _ = new_flow()
    f.start()
    run(f, "Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "consulta general", "la primera")
    assert f.state == State.READBACK and sched.appointments == {}


def test_no_record_details_before_identity_match():
    f, _, _ = new_flow()
    f.start()
    acts = run(f, "Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1981", "sí")
    assert f.state == State.NAME and "García" not in say(acts)          # misheard name: asked once more
    acts = f.handle("María Pérez")
    assert f.state == State.NEW_PATIENT and "García" not in say(acts)


def test_emergency_interrupts_any_state_and_never_reassures():
    f, sched, audit = new_flow()
    f.start()
    acts = run(f, "Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí",
               "consulta general, pero también me duele mucho el pecho")
    assert f.state == State.TRANSFERRED and f.transfer_reason == "emergency:chest_pain"
    assert "911" in say(acts) and isinstance(acts[-1], Transfer) and sched.appointments == {}
    assert any(e["kind"] == "emergency" for e in audit.events)


def test_persona_transfers_in_detected_language():
    f, _, _ = new_flow()
    f.start()
    acts = f.handle("I want to talk to a representative")
    assert f.transfer_reason == "caller_request" and "transferring you" in say(acts)


def test_two_failed_turns_transfer():
    f, _, _ = new_flow()
    f.start()
    f.handle("Hola, quiero una cita")
    assert "no le entendí" in say(f.handle(""))
    f.handle("")
    assert f.state == State.TRANSFERRED and f.transfer_reason == "not_understood"


def test_language_switch_request_repeats_prompt():
    f, _, _ = new_flow()
    f.start()
    f.handle("Hola, quiero una cita")
    acts = f.handle("Sorry, can we speak English?")
    assert f.lang == "en" and f.state == State.NAME and say(acts) == "Please tell me your full name."


def test_ambiguous_match_transfers():
    f, _, _ = new_flow()
    f.start()
    run(f, "Hola", "Me llamo Ana López", "1 de enero de 1990", "sí")
    assert f.transfer_reason == "ambiguous_match"


def test_slot_taken_reoffers_and_books_next():
    f, sched, _ = new_flow()
    f.start()
    run(f, "Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "consulta general", "la primera")
    sched.take("s1")
    acts = f.handle("sí")
    assert "acaba de ocuparse" in say(acts) and f.state == State.OFFER
    run(f, "la primera", "sí")
    assert f.booking.slot_id == "s2"


def test_disclosure_opens_with_911_in_both_languages_and_says_scheduling_only():
    f, _, _ = new_flow()
    text = say(f.start())
    parts = f.r.disclosure_parts()
    assert "911" in parts[0][0] and "911" in parts[1][0]   # the caller is deaf to us until the notice ends: 911 first
    assert "solo para citas" in text and "for scheduling only" in text
    assert text.count("911") == 2


def test_abuse_gets_a_discreet_transfer_not_the_loud_emergency_script():
    f, _, audit = new_flow()
    f.start()
    acts = f.handle("Mi esposo me pega y me amenaza")
    text = say(acts)
    assert isinstance(acts[-1], Transfer) and acts[-1].reason == "emergency:abuse"
    assert text == f.r.text("emergency_abuse", "es") and "cuelgue" not in text and "911" in text


def test_other_emergencies_keep_the_emergency_script():
    f, _, _ = new_flow()
    f.start()
    acts = f.handle("No puedo respirar")
    assert say(acts) == f.r.text("emergency", "es")
