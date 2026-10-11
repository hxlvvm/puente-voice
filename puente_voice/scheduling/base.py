"""Scheduling domain shared by every backend."""
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from puente_voice.text import fold


@dataclass(frozen=True)
class Patient:
    id: str
    name: str
    birth_date: date


@dataclass(frozen=True)
class Slot:
    id: str
    start: datetime
    end: datetime
    visit_type: str
    provider: str


@dataclass(frozen=True)
class Booking:
    appointment_id: str
    slot_id: str
    patient_id: str


class SchedulerError(Exception):
    """Backend unreachable or returned an unexpected response."""


class SlotUnavailable(SchedulerError):
    """The slot is no longer free."""


class Scheduler(Protocol):
    def find_patients(self, name: str, birth_date: date) -> list[Patient]: ...

    def search_slots(self, visit_type: str, start: datetime, end: datetime, limit: int) -> list[Slot]: ...

    def book(self, slot_id: str, patient_id: str, idempotency_key: str) -> Booking: ...


def _tokens(name: str) -> set[str]:
    return set(re.sub(r"[^a-z' \-]", " ", fold(name)).replace("-", " ").split())


def names_match(spoken: str, record: str) -> bool:
    """At least two spoken name tokens, all found in the record's name (accents and case ignored)."""
    s, r = _tokens(spoken), _tokens(record)
    return len(s) >= 2 and s <= r
