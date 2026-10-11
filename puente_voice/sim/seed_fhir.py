"""Seed a FHIR R4 server with the synthetic demo clinic (patients, practitioners, schedules, free slots).

  python -m puente_voice.sim.seed_fhir --base http://localhost:8080/fhir
"""
import argparse
import json

import httpx

from puente_voice.scheduling.fhir import VISIT_SYSTEM
from puente_voice.sim import demo_data

JSON = {"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"}


def _post(http, rtype, body):
    r = http.post(f"/{rtype}", json={"resourceType": rtype, **body})
    r.raise_for_status()
    return r.json()["id"]


def seed(base_url):
    ids = {"patients": {}, "slots": {}}
    with httpx.Client(base_url=base_url.rstrip("/"), headers=JSON, timeout=30) as http:
        for p in demo_data.patients():
            given, family = p.name.split(" ", 1) if p.name.count(" ") == 1 else (
                " ".join(p.name.split()[:2]), " ".join(p.name.split()[2:]))
            ids["patients"][p.id] = _post(http, "Patient", {
                "name": [{"given": given.split(), "family": family}], "birthDate": p.birth_date.isoformat()})
        schedules = {}
        for s in demo_data.slots():
            key = (s.provider, s.visit_type)
            if key not in schedules:
                actor = {"display": s.provider or "Enfermería"}
                schedules[key] = _post(http, "Schedule", {
                    "active": True, "actor": [actor],
                    "serviceType": [{"coding": [{"system": VISIT_SYSTEM, "code": s.visit_type}]}]})
            ids["slots"][s.id] = _post(http, "Slot", {
                "status": "free", "schedule": {"reference": f"Schedule/{schedules[key]}"},
                "serviceType": [{"coding": [{"system": VISIT_SYSTEM, "code": s.visit_type}]}],
                "start": s.start.isoformat(), "end": s.end.isoformat()})
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    print(json.dumps(seed(ap.parse_args().base), indent=1))


if __name__ == "__main__":
    main()
