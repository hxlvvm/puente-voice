"""Append-only audit events carrying IDs and enums only (never caller text, names, DOB or phone numbers)."""
import json
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = {"state", "reason", "lang", "patient_id", "slot_id", "appointment_id", "count", "ms", "version"}


class AuditLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []

    def event(self, call_id: str, kind: str, **fields):
        bad = set(fields) - ALLOWED
        if bad:
            raise ValueError(f"audit fields not allowed: {sorted(bad)}")
        for v in fields.values():
            if not isinstance(v, (str, int, float, bool, type(None))):
                raise TypeError("audit values must be scalars")
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "call_id": call_id, "kind": kind, **fields}
        self.events.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return rec
