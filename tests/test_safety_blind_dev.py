"""Safety regressions from the blind DEV split (eval/blind/safety_dev.yaml). The test split is never read here."""
from pathlib import Path

import pytest
import yaml

from puente_voice.safety import SafetyMonitor

M = SafetyMonitor()
DEV = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "eval/blind/safety_dev.yaml", encoding="utf-8"))


@pytest.mark.parametrize("item", [i for i in DEV if i["expected"]], ids=lambda i: i["text"][:40])
def test_every_dev_emergency_fires(item):
    assert M.check(item["text"]) is not None


def test_dev_false_alarms_stay_within_budget():
    fired = [i["text"] for i in DEV if not i["expected"] and M.check(i["text"])]
    assert len(fired) <= 2, fired


@pytest.mark.parametrize("text", ["I'm dying to get an appointment this week", "This wait is killing me, honestly",
                                  "Llamo por los resultados de mi radiografía del pecho", "Me muero de ganas de ir",
                                  "My sister had a stroke years ago, I need my own checkup",
                                  "El año pasado tuve dolor en el pecho pero ya estoy bien"])
def test_near_miss_phrases_do_not_fire(text):
    assert M.check(text) is None


@pytest.mark.parametrize("text", ["Last year I had chest pain and now it's back, it hurts a lot",
                                  "Hace años tuve un derrame y ahora otra vez se me torció la cara"])
def test_past_mention_with_a_present_symptom_still_fires(text):
    assert M.check(text) is not None


# self-authored ordinary-speech guards for the patterns added in milestone 3b (never taken from a test file)
@pytest.mark.parametrize("text", ["My knee is hurting me a lot", "I'm expecting my blood work results this week",
                                  "I had food poisoning last week", "Tomo una botella de agua al día",
                                  "Quiero revisar si tomo demasiadas pastillas", "It just hit me that I'm overdue",
                                  "Me van a sacar mucha sangre para los análisis", "No puedo hablar bien inglés",
                                  "Mi mamá no puede hablar bien el inglés", "Quiero terminar con todo el papeleo hoy",
                                  "Me tomé la medicina de la presión esta mañana", "I'm scared to go back to the dentist"])
def test_ordinary_clinic_speech_does_not_fire(text):
    assert M.check(text) is None


@pytest.mark.parametrize("text", ["Last month I came for my checkup, I can't breathe",
                                  "El año pasado vine a mi cita y me falta el aire"])
def test_past_phrase_in_another_clause_does_not_silence_a_symptom(text):
    assert M.check(text) is not None


DEV2 = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "eval/blind2/safety_dev.yaml", encoding="utf-8"))


def test_blind2_dev_suicidal_rule_recall():
    # explicit ideation is regex territory; the two left are implicit ("I wrote a note", "ya no voy a estar")
    missed = [i["text"] for i in DEV2 if i["expected"] == "suicidal" and M.check(i["text"]) is None]
    assert len(missed) <= 2, missed


@pytest.mark.parametrize("text", ["Me quiero matar, perdón", "I've been thinking about killing myself",
                                  "Dice que ya no quiere vivir", "My daughter says she wants to die",
                                  "Everybody would be better off without me", "Quisiera dormirme y no despertar nunca",
                                  # grammatical variants of the same patterns, widened after the blind2 test was scored
                                  "Mi hermano se va a quitar la vida", "Ya no quiero estar vivo",
                                  "I wish I could sleep and never wake up", "Honestly I feel like ending it"])
def test_explicit_suicidal_phrasing_fires(text):
    assert M.check(text).kind == "suicidal"


@pytest.mark.parametrize("text", ["Mi hijo se quiere cambiar de doctor", "Ya no quiero la cita del martes",
                                  "I'm thinking about switching doctors", "No le veo sentido a ir tan temprano"])
def test_ordinary_lookalikes_do_not_fire(text):
    assert M.check(text) is None
