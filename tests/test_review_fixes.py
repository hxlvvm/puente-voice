"""Regression tests for the milestone-1 final review findings."""
import json
from datetime import date

import httpx
import pytest

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow, State, Transfer
from puente_voice.interpret import RuleInterpreter
from puente_voice.safety import SafetyMonitor
from puente_voice.scheduling.base import SchedulerError
from puente_voice.scheduling.fhir import CALL_SYSTEM, FhirScheduler
from puente_voice.sim import demo_data

IP = RuleInterpreter(demo_data.CLINIC)
M = SafetyMonitor()
BASE = "http://fhir.test/fhir"


def flow():
    return CallFlow("call-r", demo_data.CLINIC, demo_data.scheduler(), AuditLog(), now=lambda: demo_data.NOW)


def fhir(handler):
    return FhirScheduler(BASE, client=httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE))


# Critical 1: never "yes" on a negated or conditional answer
@pytest.mark.parametrize("text", ["That's not right, I wanted Wednesday", "Sí, pero prefiero el miércoles",
                                  "OK wait let me check", "not okay", "that's not correct", "¿sí?"])
def test_readback_never_yes_on_negated_or_conditional(text):
    assert IP.yes_no(text) != "yes"


@pytest.mark.parametrize("text", ["that's not correct", "not okay", "no es correcto"])
def test_negated_yes_word_is_no(text):
    assert IP.yes_no(text) == "no"


# Critical 2: language switching
def test_negated_language_mention_means_the_other_language():
    assert IP.language("No hablo inglés") == "es" and IP.switch_request("No hablo inglés") == "es"
    assert IP.language("I don't speak Spanish") == "en"


def test_surname_english_is_not_a_switch_request():
    assert IP.switch_request("My name is John English") is None
    assert IP.name("My name is John English") == "John English"


def test_asking_for_the_current_language_counts_as_a_failed_turn():
    f = flow()
    f.start()
    f.handle("Hola, quiero una cita")
    f.handle("¿Podemos hablar en español?")
    f.handle("¿Podemos hablar en español?")
    assert f.state == State.TRANSFERRED and f.transfer_reason == "not_understood"


# Critical 3: negation scope in the safety monitor
@pytest.mark.parametrize("text,kind", [("no tengo fiebre me duele el pecho", "chest_pain"),
                                       ("no se que hacer me duele el pecho", "chest_pain"),
                                       ("no no puedo respirar", "breathing"),
                                       ("i don't know i can't breathe", "breathing"),
                                       ("I've never had chest pain like this", "chest_pain"),
                                       ("Nunca había tenido un dolor de pecho así", "chest_pain"),
                                       ("No me pasa el dolor de pecho", "chest_pain")])
def test_unpunctuated_or_distant_negation_still_fires(text, kind):
    a = M.check(text)
    assert a and a.kind == kind


@pytest.mark.parametrize("text", ["I don't have any chest pain", "No tengo ningún dolor de pecho",
                                  "no es una emergencia, solo quiero una cita", "it's not an emergency",
                                  "I haven't had chest pain"])
def test_close_negation_still_suppresses(text):
    assert M.check(text) is None


# Important: missing emergency phrases
@pytest.mark.parametrize("text,kind", [("es una emergencia", "emergency"), ("this is an emergency", "emergency"),
                                       ("creo que es un infarto", "chest_pain"),
                                       ("I think it's a heart attack", "chest_pain"),
                                       ("I think I'm having a stroke", "stroke"), ("tuvo un derrame cerebral", "stroke"),
                                       ("I'm having difficulty breathing", "breathing"),
                                       ("I can not breathe", "breathing"), ("no paro de sangrar", "bleeding")])
def test_added_emergency_phrases(text, kind):
    a = M.check(text)
    assert a and a.kind == kind


# Important: slot choice
@pytest.mark.parametrize("text", ["the one at 12:30", "the Tuesday one", "the one with Dr. Romero", "la de las 2",
                                  "a las tres"])
def test_choice_ignores_times_and_bare_one(text):
    assert IP.choice(text, 3) is None


@pytest.mark.parametrize("text,i", [("opción 2", 1), ("option 3", 2), ("2", 1), ("la segunda", 1),
                                    ("the first one", 0), ("número uno", 0)])
def test_choice_accepts_explicit_options(text, i):
    assert IP.choice(text, 3) == i


# Important: name parser
@pytest.mark.parametrize("text,name", [("Mi nombre completo es María Pérez", "María Pérez"),
                                       ("My full name is John Smith", "John Smith"),
                                       ("Sí, me llamo María Pérez", "María Pérez"),
                                       ("Buenos días, me llamo María Pérez", "María Pérez"),
                                       ("Quiero una cita", None), ("Sorry can you repeat that", None)])
def test_name_strips_fillers_and_rejects_requests(text, name):
    assert IP.name(text) == name


def test_no_match_reasks_the_name_once():
    f = flow()
    f.start()
    for t in ["Hola, quiero una cita", "Me llamo Mariana Pérez", "15 de marzo de 1980", "sí"]:
        acts = f.handle(t)
    assert f.state == State.NAME and "repetir su nombre" in acts[0].text
    f.handle("María Pérez")
    assert f.state == State.VISIT and f.patient.id == "p1"


# Important: safety after the goodbye
def test_safety_still_runs_after_booking():
    f = flow()
    f.start()
    for t in ["Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "consulta general",
              "la primera", "sí"]:
        f.handle(t)
    assert f.state == State.DONE
    acts = f.handle("espere, me duele mucho el pecho")
    assert isinstance(acts[-1], Transfer) and f.transfer_reason == "emergency:chest_pain"


# Re-graded: visit-type collision
def test_consulta_de_seguimiento_is_follow_up():
    assert IP.visit("consulta de seguimiento") == "follow-up"


# Important: FHIR patient search
def test_patient_search_keeps_dob_out_of_the_url():
    def h(req):
        assert req.method == "POST" and req.url.path.endswith("/Patient/_search")
        assert "1980" not in str(req.url) and b"birthdate=1980-03-15" in req.content
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": []})
    assert fhir(h).find_patients("Maria Perez", date(1980, 3, 15)) == []


def test_patient_search_with_more_pages_is_an_error():
    def h(req):
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [],
                                         "link": [{"relation": "next", "url": f"{BASE}?page=2"}]})
    with pytest.raises(SchedulerError):
        fhir(h).find_patients("Maria Perez", date(1980, 3, 15))


# Important: malformed FHIR replies become SchedulerError, a lost reply is recovered
def _slot(meta=True):
    s = {"resourceType": "Slot", "id": "s1", "status": "free", "schedule": {"reference": "Schedule/sc1"},
         "start": "2026-10-13T15:00:00Z", "end": "2026-10-13T15:30:00Z"}
    return {**s, "meta": {"versionId": "3"}} if meta else s


def test_slot_without_version_is_a_scheduler_error():
    def h(req):
        if req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json={"resourceType": "Bundle"})
        return httpx.Response(200, json=_slot(meta=False))
    with pytest.raises(SchedulerError):
        fhir(h).book("s1", "p1", "call-1")


def test_unreadable_transaction_reply_recovers_via_idempotent_lookup():
    lookups = []

    def h(req):
        if req.url.path.endswith("/Appointment"):
            lookups.append(1)
            if len(lookups) == 1:
                return httpx.Response(200, json={"resourceType": "Bundle"})
            return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": {
                "resourceType": "Appointment", "id": "a9", "slot": [{"reference": "Slot/s1"}],
                "participant": [{"actor": {"reference": "Patient/p1"}}]}}]})
        if req.method == "GET":
            return httpx.Response(200, json=_slot())
        return httpx.Response(200, content=b"")
    assert fhir(h).book("s1", "p1", "call-1").appointment_id == "a9"


# Re-graded: Provenance must carry the call ID (spec section 6)
def test_provenance_carries_the_call_id():
    seen = {}

    def h(req):
        if req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json={"resourceType": "Bundle"})
        if req.method == "GET":
            return httpx.Response(200, json=_slot())
        seen["tx"] = json.loads(req.content)
        return httpx.Response(200, json={"entry": [{"response": {"location": "Slot/s1/_history/4"}},
                                                   {"response": {"location": "Appointment/a7/_history/1"}},
                                                   {"response": {"location": "Provenance/p/_history/1"}}]})
    fhir(h).book("s1", "p1", "call-1")
    prov = seen["tx"]["entry"][2]["resource"]
    assert prov["entity"][0]["what"]["identifier"] == {"system": CALL_SYSTEM, "value": "call-1"}
