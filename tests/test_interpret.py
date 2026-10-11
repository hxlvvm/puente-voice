from datetime import date
from pathlib import Path

import pytest

from puente_voice.config import load_clinic
from puente_voice.interpret import RuleInterpreter

IP = RuleInterpreter(load_clinic(Path(__file__).resolve().parent.parent / "config/clinic.example.yaml"))


@pytest.mark.parametrize("text,lang", [("Hola, quiero una cita", "es"), ("Español, por favor", "es"),
                                       ("Hello, I need an appointment", "en"), ("English", "en"), ("Okay", None)])
def test_language(text, lang):
    assert IP.language(text) == lang


def test_switch_request_needs_explicit_word():
    assert IP.switch_request("Sorry, can we speak English?") == "en"
    assert IP.switch_request("¿Podemos hablar en español?") == "es"
    assert IP.switch_request("My name is John Smith") is None


@pytest.mark.parametrize("text,ans", [("Sí, por favor", "yes"), ("sí, correcto", "yes"), ("Yes, that's right", "yes"),
                                      ("No, otra", "no"), ("nope", "no"), ("creo que sí", "unclear"),
                                      ("maybe", "unclear"), ("sí, pero no el martes", "unclear"), ("", "unclear")])
def test_yes_no(text, ans):
    assert IP.yes_no(text) == ans


def test_persona():
    assert IP.persona("quiero hablar con una persona") and IP.persona("Representative, please")
    assert not IP.persona("quiero una cita")


@pytest.mark.parametrize("text,name", [("Me llamo María Pérez.", "María Pérez"), ("My name is John Smith", "John Smith"),
                                       ("María José Pérez García", "María José Pérez García"), ("Juan", None),
                                       ("sí", None)])
def test_name(text, name):
    assert IP.name(text) == name


@pytest.mark.parametrize("text,lang,d", [("15 de marzo de 1980", "es", date(1980, 3, 15)),
                                         ("el primero de enero de 1990", "es", date(1990, 1, 1)),
                                         ("July 4th, 1975", "en", date(1975, 7, 4)),
                                         ("the 4th of July 1975", "en", date(1975, 7, 4)),
                                         ("31 de febrero de 1980", "es", None), ("no sé", "es", None)])
def test_dob(text, lang, d):
    assert IP.dob(text, lang) == d


def test_numeric_dob_depends_on_language():
    assert IP.dob("03/04/1980", "es") == date(1980, 4, 3)
    assert IP.dob("03/04/1980", "en") == date(1980, 3, 4)


@pytest.mark.parametrize("text,code", [("una consulta general", "general"), ("a checkup", "general"),
                                       ("seguimiento de resultados", "follow-up"), ("I need a flu shot", "vaccine"),
                                       ("no sé", None)])
def test_visit(text, code):
    assert IP.visit(text) == code


@pytest.mark.parametrize("text,n,i", [("la primera", 3, 0), ("the second one", 3, 1), ("3", 3, 2), ("la última", 3, 2),
                                      ("la cuarta", 3, None), ("la primera o la segunda", 3, None)])
def test_choice(text, n, i):
    assert IP.choice(text, n) == i


def test_phone():
    assert IP.phone("312 555 0199") == "3125550199"
    assert IP.phone("1-312-555-0199") == "3125550199"
    assert IP.phone("555 0199") is None


@pytest.mark.parametrize("text", ["Gracias.", "Gracias por ver el video.", "¡Suscríbete!", "Thank you for watching."])
def test_whisper_hallucinations_are_never_a_yes_or_a_name(text):
    """Whisper's stock outputs for line noise; measured on noise and a cough at 8 kHz (job 7571432 and after)."""
    assert IP.yes_no(text) != "yes" and IP.name(text) is None
