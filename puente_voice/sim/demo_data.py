"""Synthetic clinic, patients and slots for tests and the simulator. No real people."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from puente_voice.config import load_clinic
from puente_voice.scheduling.base import Patient, Slot
from puente_voice.scheduling.memory import InMemoryScheduler

CLINIC = load_clinic(Path(__file__).resolve().parents[2] / "config/clinic.example.yaml")
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def patients():
    return [Patient("p1", "María José Pérez García", date(1980, 3, 15)),
            Patient("p2", "John Smith", date(1975, 7, 4)),
            Patient("p3", "Ana López", date(1990, 1, 1)),
            Patient("p4", "Ana López", date(1990, 1, 1))]


def slots():
    def s(i, start, kind, provider):
        return Slot(i, start, start + timedelta(minutes=30), kind, provider)
    d = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
    return [s("s1", d, "general", "Dra. Ana García"),
            s("s2", d + timedelta(hours=2, minutes=30), "general", "Dra. Ana García"),
            s("s3", d + timedelta(days=1, hours=-1), "general", "Dr. Luis Romero"),
            s("s4", d + timedelta(days=2), "follow-up", "Dra. Ana García"),
            s("s5", d + timedelta(days=1), "vaccine", "")]


def scheduler():
    return InMemoryScheduler(patients(), slots())
