import pytest

from puente_voice.safety import SafetyMonitor
from puente_voice.text import fold

M = SafetyMonitor()


def kind(t):
    a = M.check(t)
    return a.kind if a else None


def test_fold():
    assert fold("  Sí, me DUELE   el Pecho ") == "si, me duele el pecho"
    assert fold("I'm fine") == "i'm fine"


@pytest.mark.parametrize("text,expected", [
    ("Me duele mucho el pecho", "chest_pain"),
    ("tengo una presión en el pecho", "chest_pain"),
    ("I have chest pain since this morning", "chest_pain"),
    ("no puedo respirar bien", "breathing"),
    ("I'm short of breath", "breathing"),
    ("se me cayó la cara de un lado", "stroke"),
    ("he pensado en quitarme la vida", "suicidal"),
    ("I want to kill myself", "suicidal"),
    ("mi hijo se tomó todas las pastillas", "overdose"),
    ("estoy embarazada y tengo sangrado", "pregnancy_bleeding"),
    ("I'm bleeding heavily", "bleeding"),
    ("mi esposo me pega", "abuse"),
])
def test_positives(text, expected):
    assert kind(text) == expected


@pytest.mark.parametrize("text", [
    "no tengo dolor de pecho",
    "No tengo dolor de pecho ni falta de aire",
    "I don't have chest pain",
    "quiero una cita para mi chequeo",
    "My name is John Smith",
])
def test_negated_or_absent(text):
    assert kind(text) is None


def test_negation_only_within_clause():
    assert kind("no tengo fiebre pero me duele el pecho") == "chest_pain"
    assert kind("No tengo fiebre, me duele el pecho") == "chest_pain"
    assert kind("I don't have a fever but I have chest pain") == "chest_pain"


def test_suicidal_fires_even_when_negated():
    assert kind("no, no he pensado en quitarme la vida") == "suicidal"


def test_both_languages_checked_regardless_of_call_language():
    assert kind("Hola, tengo chest pain") == "chest_pain"
