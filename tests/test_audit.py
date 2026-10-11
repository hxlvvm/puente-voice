import json

import pytest

from puente_voice.audit import AuditLog


def test_records_allowed_fields_and_writes_jsonl(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    log.event("c1", "booked", appointment_id="a1", slot_id="s1")
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["call_id"] == "c1" and rec["kind"] == "booked" and rec["slot_id"] == "s1" and "ts" in rec
    assert log.events[0]["appointment_id"] == "a1"


@pytest.mark.parametrize("field", ["text", "name", "dob", "phone", "utterance"])
def test_rejects_fields_that_could_hold_patient_data(field):
    with pytest.raises(ValueError):
        AuditLog().event("c1", "x", **{field: "María Pérez"})


def test_rejects_non_scalar_values():
    with pytest.raises(TypeError):
        AuditLog().event("c1", "x", reason={"nested": "value"})
