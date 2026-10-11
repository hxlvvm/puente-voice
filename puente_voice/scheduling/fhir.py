"""FHIR R4 scheduler: patient match by birth date + name tokens, free Slot search, transactional booking."""
import uuid
from datetime import date, datetime, timezone

import httpx

from puente_voice import __version__
from puente_voice.scheduling.base import Booking, Patient, SchedulerError, Slot, SlotUnavailable, names_match

VISIT_SYSTEM = "urn:puente-voice:visit-type"
CALL_SYSTEM = "urn:puente-voice:call-id"
JSON = "application/fhir+json"
NO_CACHE = {"Cache-Control": "no-cache"}  # HAPI reuses cached search results for ~60 s by default
PARSE_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _id(location: str) -> str:
    return location.split("/")[1]


class FhirScheduler:
    def __init__(self, base_url, client=None, timeout=10.0):
        self.http = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers={"Accept": JSON})

    def _send(self, method, path, **kw):
        try:
            r = self.http.request(method, path, headers=NO_CACHE, **kw)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise SchedulerError(f"{method} {path} failed: {type(e).__name__}") from None

    def _get(self, path, **params):
        return self._send("GET", path, params=params)

    def find_patients(self, name, birth_date):
        # POST _search keeps the date of birth out of URLs, proxy logs and client request logs
        b = self._send("POST", "/Patient/_search", data={"birthdate": birth_date.isoformat(), "_count": "50"})
        if any(link.get("relation") == "next" for link in b.get("link", [])):
            raise SchedulerError("patient search has more pages")  # the exactly-one rule needs every match
        try:
            out = []
            for e in b.get("entry", []):
                p = e["resource"]
                n = (p.get("name") or [{}])[0]
                full = " ".join(n.get("given", []) + [n.get("family", "")]).strip()
                if names_match(name, full):
                    out.append(Patient(p["id"], full, date.fromisoformat(p["birthDate"])))
            return out
        except PARSE_ERRORS:
            raise SchedulerError("unexpected Patient bundle") from None

    def search_slots(self, visit_type, start, end, limit):
        b = self._get("/Slot", status="free", **{"service-type": f"{VISIT_SYSTEM}|{visit_type}"},
                      start=[f"ge{start.isoformat()}", f"lt{end.isoformat()}"], _sort="start", _count=limit,
                      _include="Slot:schedule")
        try:
            res = [e["resource"] for e in b.get("entry", [])]
            providers = {f"Schedule/{r['id']}": (r.get("actor") or [{}])[0].get("display", "")
                         for r in res if r["resourceType"] == "Schedule"}
            slots = [Slot(r["id"], _dt(r["start"]), _dt(r["end"]), visit_type,
                          providers.get(r["schedule"]["reference"], ""))
                     for r in res if r["resourceType"] == "Slot"]
        except PARSE_ERRORS:
            raise SchedulerError("unexpected Slot bundle") from None
        return sorted(slots, key=lambda s: s.start)[:limit]

    def _existing(self, key):
        b = self._get("/Appointment", identifier=f"{CALL_SYSTEM}|{key}")
        if not b.get("entry"):
            return None
        try:
            a = b["entry"][0]["resource"]
            return Booking(a["id"], a["slot"][0]["reference"].split("/")[1],
                           a["participant"][0]["actor"]["reference"].split("/")[1])
        except PARSE_ERRORS:
            raise SchedulerError("unexpected Appointment bundle") from None

    def book(self, slot_id, patient_id, idempotency_key):
        found = self._existing(idempotency_key)
        if found:
            return found
        slot = self._get(f"/Slot/{slot_id}")
        if slot.get("status") != "free":
            raise SlotUnavailable(slot_id)
        try:
            version = slot["meta"]["versionId"]  # no version, no safe compare-and-set: refuse rather than risk it
            appt_url, now = f"urn:uuid:{uuid.uuid4()}", datetime.now(timezone.utc).isoformat()
            busy = {k: v for k, v in slot.items() if k != "meta"} | {"status": "busy"}
            tx = {"resourceType": "Bundle", "type": "transaction", "entry": [
                {"resource": busy, "request": {"method": "PUT", "url": f"Slot/{slot_id}", "ifMatch": f'W/"{version}"'}},
                {"fullUrl": appt_url, "request": {"method": "POST", "url": "Appointment"},
                 "resource": {"resourceType": "Appointment", "status": "booked",
                              "identifier": [{"system": CALL_SYSTEM, "value": idempotency_key}],
                              "serviceType": slot.get("serviceType", []), "slot": [{"reference": f"Slot/{slot_id}"}],
                              "start": slot["start"], "end": slot["end"],
                              "participant": [{"actor": {"reference": f"Patient/{patient_id}"},
                                               "status": "accepted"}]}},
                {"request": {"method": "POST", "url": "Provenance"},
                 "resource": {"resourceType": "Provenance", "target": [{"reference": appt_url}], "recorded": now,
                              "activity": {"text": "Booked by voice agent; patient confirmed by read-back"},
                              "agent": [{"who": {"display": f"puente-voice {__version__}"}}],
                              "entity": [{"role": "source", "what": {"identifier": {
                                  "system": CALL_SYSTEM, "value": idempotency_key}}}]}}]}
        except PARSE_ERRORS:
            raise SchedulerError("unexpected Slot resource") from None
        try:
            r = self.http.post("", json=tx, headers={"Content-Type": JSON})
        except httpx.HTTPError:
            raise SchedulerError("transaction failed") from None
        if r.status_code in (409, 412):
            raise SlotUnavailable(slot_id)
        if r.status_code >= 400:
            raise SchedulerError(f"transaction returned {r.status_code}")
        try:
            return Booking(_id(r.json()["entry"][1]["response"]["location"]), slot_id, patient_id)
        except PARSE_ERRORS:
            # committed but the reply was unreadable: the idempotency lookup tells whether it was booked
            found = self._existing(idempotency_key)
            if found:
                return found
            raise SchedulerError("transaction reply unreadable") from None
