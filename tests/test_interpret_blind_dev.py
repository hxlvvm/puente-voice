"""Interpreter regressions from the blind DEV split (eval/blind/meaning_dev.yaml). The test split is never read here."""
from datetime import date
from pathlib import Path

import pytest

from puente_voice.config import load_clinic
from puente_voice.interpret import RuleInterpreter

IP = RuleInterpreter(load_clinic(Path(__file__).resolve().parent.parent / "config/clinic.example.yaml"))


@pytest.mark.parametrize("text,lang,want", [
    ("Nací el quince de marzo de mil novecientos ochenta.", "es", date(1980, 3, 15)),
    ("Este... el 3 de julio del 92.", "es", date(1992, 7, 3)),
    ("Pues el primero de enero de dos mil cuatro.", "es", date(2004, 1, 1)),
    ("Ay, el veintiocho de febrero de mil novecientos sesenta y ocho.", "es", date(1968, 2, 28)),
    ("O sea, doce de diciembre del dos mil.", "es", date(2000, 12, 12)),
    ("Nací el treinta de abril del setenta y tres.", "es", date(1973, 4, 30)),
    ("So, um, the third of October, nineteen sixty-five.", "en", date(1965, 10, 3)),
    ("December 1st, two thousand two.", "en", date(2002, 12, 1)),
    ("Yeah, July twenty-ninth, ninety-one.", "en", date(1991, 7, 29)),
    ("el 3 de julio del 05", "es", date(2005, 7, 3)),       # two-digit years: < 30 -> 20xx
    ("el 3 de julio del 29", "es", None),                    # ... and a future date is no date
])
def test_spoken_dates(text, lang, want):
    assert IP.dob(text, lang) == want


@pytest.mark.parametrize("text,want", [
    ("Sí, mire, me llamo María Fernanda López Ramírez.", "María Fernanda López Ramírez"),
    ("Este... Juan Carlos Hernández.", "Juan Carlos Hernández"),
    ("Ay, es Guadalupe Sánchez Ortiz, para servirle.", "Guadalupe Sánchez Ortiz"),
    ("O sea, Ana Lucía Domínguez.", "Ana Lucía Domínguez"),
    ("Mire, yo soy Miguel Ángel Rodríguez Cruz.", "Miguel Ángel Rodríguez Cruz"),
    ("Pues Carmen Rivera de Jesús, señorita.", "Carmen Rivera de Jesús"),
    ("Eh, me llamo Édgar Giovanni Mejía.", "Édgar Giovanni Mejía"),
    ("Uh, yeah, it's Michael James Thompson.", "Michael James Thompson"),
    ("So, um, Robert Alvarez.", "Robert Alvarez"),
    ("This is Jennifer O'Connor calling.", "Jennifer O'Connor"),
    ("Um, Patricia Williams-Brown.", "Patricia Williams-Brown"),
    ("Habla Pedro Ramírez", "Pedro Ramírez"),                # M3a miss, listed in the plan
])
def test_names_with_fillers_and_courtesy(text, want):
    assert IP.name(text) == want


@pytest.mark.parametrize("text,want", [
    ("Me llamo Pedro Pues", "Pedro Pues"),                 # a surname that is also a filler stays
    ("Bueno, me llamo Bueno García", "Bueno García"),
    ("Mi nombre es Ana Este", "Ana Este"),
])
def test_fillers_only_before_the_name(text, want):
    assert IP.name(text) == want


@pytest.mark.parametrize("text,want", [("Este... la del medio.", 1), ("Uno, la uno.", 0), ("The middle one", 1),
                                       ("Un segundo, déjeme ver", None), ("Wait a second", None),
                                       ("Un momento por favor", None), ("La segunda, por favor", 1)])
def test_choice(text, want):
    assert IP.choice(text, 3) == want


@pytest.mark.parametrize("text", ["Ay, no, yo quiero que me atienda alguien de verdad.",
                                  "¿Me puede comunicar con la recepcionista?", "Can I talk to the front desk?"])
def test_persona_requests(text):
    assert IP.persona(text) is True


@pytest.mark.parametrize("text,want", [
    ("Es para el examen físico de mi trabajo.", "general"),
    ("Pues quiero que el doctor me revise, hace tiempo que no voy.", "general"),
    ("Ay, para mi consulta de rutina, este, la de cada año.", "general"),
    ("Pues mi niño necesita las vacunas para la escuela.", "vaccine"),
    ("I need my annual physical for work.", "general"),
    ("Yeah, I haven't seen a doctor in a while, so just a routine visit.", "general"),
    ("So my daughter is due for her booster shots.", "vaccine"),
])
def test_visit_phrasings(text, want):
    assert IP.visit(text) == want


@pytest.mark.parametrize("text,want", [("No, mejor no, gracias.", "no"), ("Vale", "yes"), ("Eso es", "yes"),
                                       ("Ándale, sí", "yes")])
def test_yes_no_dev(text, want):
    assert IP.yes_no(text) == want


@pytest.mark.parametrize("text", ["Vale, pero espere", "Eso es lo que no sé", "Ándale, pero mejor el martes",
                                  "Vale, creo que sí", "Mejor sí, ¿no?"])
def test_new_yes_words_keep_conditionals_unclear(text):
    assert IP.yes_no(text) == "unclear"


@pytest.mark.parametrize("text", ["La recepcionista me dijo que llamara", "The appointment is for a person in my family",
                                  "The front desk gave me this number"])
def test_mentioning_staff_is_not_a_transfer_request(text):
    assert IP.persona(text) is False
