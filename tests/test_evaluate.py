from pathlib import Path

import pytest

from puente_voice.evaluate import load_set, meaning, report, score, wilson
from puente_voice.interpret import RuleInterpreter
from puente_voice.safety import RULES
from puente_voice.sim import demo_data

IP = RuleInterpreter(demo_data.CLINIC)
EVAL = Path(__file__).resolve().parent.parent / "eval"


def test_meaning_dispatches_to_the_interpreter():
    assert meaning(IP, "name", "Me llamo María Pérez", "es") == "María Pérez"
    assert meaning(IP, "dob", "15 de marzo de 1980", "es") == "1980-03-15"
    assert meaning(IP, "visit", "consulta de seguimiento", "es") == "follow-up"
    assert meaning(IP, "choice", "la segunda", "es") == 1
    assert meaning(IP, "yes_no", "sí, correcto", "es") == "yes"
    assert meaning(IP, "language", "Hello, I need an appointment", "en") == "en"
    assert meaning(IP, "persona", "quiero hablar con una persona", "es") is True


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.237) < 0.005 and abs(hi - 0.763) < 0.005
    assert wilson(10, 10)[1] == pytest.approx(1.0) and wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    assert wilson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("name", ["utterances.yaml", "safety_set.yaml"])
def test_corpora_are_well_formed_and_split(name):
    items = load_set(EVAL / name)
    keys = {"kind", "lang", "text", "expected", "split"} if name == "utterances.yaml" else {"lang", "text", "expected", "split"}
    assert all(keys <= set(i) and i["split"] in ("dev", "test") and i["lang"] in ("es", "en") for i in items)
    dev = {i["text"] for i in items if i["split"] == "dev"}
    test = {i["text"] for i in items if i["split"] == "test"}
    assert dev and test and not dev & test
    if name == "utterances.yaml":
        for kind in {i["kind"] for i in items}:
            assert {i["split"] for i in items if i["kind"] == kind} == {"dev", "test"}, kind
    else:
        assert {i["expected"] for i in items} - {None} <= set(RULES)


def test_score_and_report_count_misses():
    items = [{"kind": "yes_no", "lang": "es", "text": "sí", "expected": "yes", "split": "dev"},
             {"kind": "yes_no", "lang": "es", "text": "simón", "expected": "yes", "split": "dev"}]
    s = score(items, IP, heard=lambda it: "See you." if it["text"] == "sí" else it["text"])
    assert s["yes_no"][:2] == [0, 2] and len(s["yes_no"][2]) == 2
    assert report(s)[0].startswith("yes_no")


def test_phone_segment_has_the_pipeline_shape():
    pytest.importorskip("pipecat")
    import numpy as np

    from puente_voice.evaluate import phone_segment
    one_second = (np.sin(np.arange(24000) / 24000 * 2 * np.pi * 300) * 8000).astype(np.int16).tobytes()
    x = phone_segment(one_second, 24000)
    assert x.dtype == np.float32 and abs(len(x) - 16000 * 2.5) <= 32


def test_safety_scores_count_recall_and_false_alarms():
    from puente_voice.evaluate import safety_scores
    from puente_voice.safety import SafetyMonitor
    items = [{"lang": "es", "text": "me duele mucho el pecho", "expected": "chest_pain", "split": "dev"},
             {"lang": "es", "text": "no tengo dolor de pecho", "expected": None, "split": "dev"},
             {"lang": "en", "text": "I can't breathe", "expected": "breathing", "split": "dev"}]
    s = safety_scores(items, SafetyMonitor(), heard=lambda it: "" if "breathe" in it["text"] else it["text"])
    assert s["recall"] == (1, 2) and s["false_alarms"] == (0, 1) and len(s["misses"]) == 1


def test_voice_choice_ignores_si():
    from puente_voice.evaluate import choose_voice
    real = {"hola": 0.9, "claro": 0.96, "sí": 0.0}
    voices = {"a": {"hola": 0.33, "claro": 1.0, "sí": 0.0}, "b": {"hola": 0.9, "claro": 0.9, "sí": 1.0}}
    assert choose_voice(real, voices) == "b"
    assert choose_voice(real, {"a": voices["a"], "c": {"hola": 0.3, "claro": 1.0, "sí": 1.0}}) == "a"  # 0.305 < 0.32
