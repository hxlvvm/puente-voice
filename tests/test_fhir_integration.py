"""Runs only with a live FHIR server: PUENTE_FHIR_URL=http://localhost:8080/fhir pytest -m fhir"""
import os
from datetime import date

import pytest

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow, State
from puente_voice.scheduling.base import SlotUnavailable
from puente_voice.scheduling.fhir import FhirScheduler
from puente_voice.sim import demo_data
from puente_voice.sim.seed_fhir import seed

URL = os.environ.get("PUENTE_FHIR_URL")
pytestmark = [pytest.mark.fhir, pytest.mark.skipif(not URL, reason="PUENTE_FHIR_URL not set")]


@pytest.fixture(scope="module")
def ids():
    return seed(URL)


def test_happy_call_books_on_the_server(ids):
    sched = FhirScheduler(URL)
    f = CallFlow("it-call-1", demo_data.CLINIC, sched, AuditLog(), now=lambda: demo_data.NOW)
    f.start()
    for t in ["Hola, quiero una cita", "Me llamo María Pérez", "15 de marzo de 1980", "sí", "vacuna", "la primera",
              "sí"]:
        f.handle(t)
    assert f.state == State.DONE and f.booking.patient_id == ids["patients"]["p1"]
    again = sched.book(f.booking.slot_id, f.booking.patient_id, "it-call-1")
    assert again.appointment_id == f.booking.appointment_id
    with pytest.raises(SlotUnavailable):
        sched.book(f.booking.slot_id, ids["patients"]["p2"], "it-call-2")


def test_patient_match_is_exactly_one(ids):
    got = FhirScheduler(URL).find_patients("Maria Perez", date(1980, 3, 15))
    assert [p.id for p in got] == [ids["patients"]["p1"]]
