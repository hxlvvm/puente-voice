import json
from datetime import date, datetime, timezone

import httpx
import pytest

from puente_voice.scheduling.base import SchedulerError, SlotUnavailable
from puente_voice.scheduling.fhir import CALL_SYSTEM, FhirScheduler

BASE = "http://fhir.test/fhir"


def bundle(*resources):
    return {"resourceType": "Bundle", "type": "searchset", "entry": [{"resource": r} for r in resources]}


def patient(pid, given, family, dob):
    return {"resourceType": "Patient", "id": pid, "birthDate": dob, "name": [{"given": given, "family": family}]}


def make(handler):
    return FhirScheduler(BASE, client=httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE))


def test_find_patients_filters_by_name_tokens():
    def h(req):
        assert req.url.path.endswith("/Patient/_search") and b"birthdate=1980-03-15" in req.content
        return httpx.Response(200, json=bundle(patient("p1", ["María", "José"], "Pérez García", "1980-03-15"),
                                               patient("p9", ["María"], "Pérez", "1980-03-15")))
    got = make(h).find_patients("Maria Jose Perez", date(1980, 3, 15))
    assert [p.id for p in got] == ["p1"] and got[0].birth_date == date(1980, 3, 15)


def test_search_slots_maps_provider_from_included_schedule():
    def h(req):
        p = req.url.params
        assert p["status"] == "free" and p["service-type"].endswith("|general") and p["_include"] == "Slot:schedule"
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [
            {"resource": {"resourceType": "Slot", "id": "s1", "schedule": {"reference": "Schedule/sc1"},
                          "start": "2026-10-13T15:00:00Z", "end": "2026-10-13T15:30:00Z",
                          "serviceType": [{"coding": [{"code": "general"}]}]}},
            {"resource": {"resourceType": "Schedule", "id": "sc1",
                          "actor": [{"reference": "Practitioner/pr1", "display": "Dra. Ana García"}]}}]})
    got = make(h).search_slots("general", datetime(2026, 10, 12, tzinfo=timezone.utc),
                               datetime(2026, 10, 26, tzinfo=timezone.utc), 3)
    assert got[0].id == "s1" and got[0].provider == "Dra. Ana García"
    assert got[0].start == datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)


def _slot(status="free"):
    return {"resourceType": "Slot", "id": "s1", "status": status, "meta": {"versionId": "3"},
            "schedule": {"reference": "Schedule/sc1"}, "start": "2026-10-13T15:00:00Z", "end": "2026-10-13T15:30:00Z",
            "serviceType": [{"coding": [{"code": "general"}]}]}


def test_book_sends_one_transaction_with_version_check_and_provenance():
    seen = {}

    def h(req):
        if req.method == "GET" and req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json=bundle())
        if req.method == "GET" and req.url.path.endswith("/Slot/s1"):
            return httpx.Response(200, json=_slot())
        seen["tx"] = json.loads(req.content)
        return httpx.Response(200, json={"resourceType": "Bundle", "type": "transaction-response", "entry": [
            {"response": {"status": "200 OK", "location": "Slot/s1/_history/4"}},
            {"response": {"status": "201 Created", "location": "Appointment/a7/_history/1"}},
            {"response": {"status": "201 Created", "location": "Provenance/pv1/_history/1"}}]})
    b = make(h).book("s1", "p1", "call-1")
    assert b.appointment_id == "a7" and b.slot_id == "s1" and b.patient_id == "p1"
    entries = seen["tx"]["entry"]
    assert entries[0]["request"] == {"method": "PUT", "url": "Slot/s1", "ifMatch": 'W/"3"'}
    assert entries[0]["resource"]["status"] == "busy"
    appt = entries[1]["resource"]
    assert appt["status"] == "booked" and appt["identifier"][0] == {"system": CALL_SYSTEM, "value": "call-1"}
    assert entries[2]["resource"]["target"][0]["reference"] == entries[1]["fullUrl"]


def test_book_is_idempotent_on_call_id():
    def h(req):
        assert req.method == "GET" and req.url.params["identifier"] == f"{CALL_SYSTEM}|call-1"
        return httpx.Response(200, json=bundle({"resourceType": "Appointment", "id": "a7",
                                                "slot": [{"reference": "Slot/s1"}],
                                                "participant": [{"actor": {"reference": "Patient/p1"}}]}))
    assert make(h).book("s1", "p1", "call-1").appointment_id == "a7"


@pytest.mark.parametrize("status", [409, 412])
def test_conflicting_write_means_slot_unavailable(status):
    def h(req):
        if req.method == "GET" and req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json=bundle())
        if req.method == "GET":
            return httpx.Response(200, json=_slot())
        return httpx.Response(status, json={"resourceType": "OperationOutcome"})
    with pytest.raises(SlotUnavailable):
        make(h).book("s1", "p1", "call-1")


def test_busy_slot_is_unavailable_and_server_error_is_scheduler_error():
    def busy(req):
        if req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json=bundle())
        return httpx.Response(200, json=_slot("busy"))
    with pytest.raises(SlotUnavailable):
        make(busy).book("s1", "p1", "call-1")
    with pytest.raises(SchedulerError):
        make(lambda req: httpx.Response(500)).find_patients("Maria Perez", date(1980, 3, 15))


def test_every_search_bypasses_the_server_search_cache():
    # HAPI reuses cached search results (~60 s by default); a stale idempotency lookup or slot list is wrong here
    seen = []

    def h(req):
        if req.method == "GET":
            seen.append(req.headers.get("cache-control"))
        if req.url.path.endswith("/Appointment"):
            return httpx.Response(200, json=bundle())
        return httpx.Response(200, json=_slot("busy") if "/Slot/" in req.url.path else bundle())
    s = make(h)
    s.find_patients("Maria Perez", date(1980, 3, 15))
    s.search_slots("general", datetime(2026, 10, 12, tzinfo=timezone.utc), datetime(2026, 10, 26, tzinfo=timezone.utc), 3)
    with pytest.raises(SlotUnavailable):
        s.book("s1", "p1", "call-1")
    assert seen and all(v == "no-cache" for v in seen)
