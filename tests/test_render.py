from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from puente_voice.config import load_clinic
from puente_voice.render import Renderer

CLINIC = load_clinic(Path(__file__).resolve().parent.parent / "config/clinic.example.yaml")
R = Renderer(CLINIC)


@dataclass
class S:
    start: datetime
    provider: str = "Dra. Ana García"


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_config_loads_and_rejects_bad_timezone():
    assert CLINIC.visit("general").es == "consulta general"
    with pytest.raises(Exception):
        type(CLINIC).model_validate({**CLINIC.model_dump(), "timezone": "Mars/Base"})


def test_dates_in_both_languages():
    assert R.date(date(1980, 3, 15), "es") == "15 de marzo de 1980"
    assert R.date(date(1980, 3, 15), "en") == "March 15, 1980"


def test_slot_time_follows_dst():
    # US DST ends 2026-11-01: 15:00 UTC is 10:00 CDT on Oct 30 and 9:00 CST on Nov 2
    assert R.slot(S(utc(2026, 10, 30, 15, 0)), "en") == "Friday, October 30 at 10:00 AM with Dra. Ana García"
    assert R.slot(S(utc(2026, 11, 2, 15, 0)), "es") == "el lunes 2 de noviembre a las 9:00 de la mañana con Dra. Ana García"


def test_spanish_time_of_day_words():
    assert "a la 1:30 de la tarde" in R.slot(S(utc(2026, 10, 13, 18, 30)), "es")
    assert "a las 12:00 del mediodía" in R.slot(S(utc(2026, 10, 13, 17, 0)), "es")
    assert "a las 7:00 de la noche" in R.slot(S(utc(2026, 10, 14, 0, 0)), "es")


def test_offer_numbers_options_and_readback_names_visit():
    slots = [S(utc(2026, 10, 13, 15, 0)), S(utc(2026, 10, 13, 17, 30))]
    es = R.offer(slots, "es")
    assert es.startswith("Tengo estas opciones: primera opción, el martes 13 de octubre a las 10:00 de la mañana")
    assert "segunda opción, el martes 13 de octubre a las 12:30 del mediodía" in es
    assert R.readback(slots[0], "general", "en").endswith("for a general visit. Shall I book it?")


def test_visit_options_and_bilingual_disclosure():
    assert R.visit_options("es") == "consulta general, cita de seguimiento o vacuna"
    assert R.visit_options("en") == "general visit, follow-up visit, or vaccination"
    d = R.disclosure()
    assert "asistente automático" in d and "automated assistant" in d and "persona" in d and "representative" in d


def test_every_template_exists_in_both_languages():
    from puente_voice.render import TEMPLATES
    for key, t in TEMPLATES.items():
        assert set(t) == {"es", "en"}, key


def test_disclosure_parts_are_monolingual_and_rejoin_to_the_disclosure():
    parts = R.disclosure_parts()
    assert [lang for _, lang in parts] == ["es", "en", "es", "en", "es", "en"]
    assert " ".join(text for text, _ in parts) == R.disclosure()
    assert "Thank you" not in parts[2][0] and "Gracias" not in parts[3][0]
