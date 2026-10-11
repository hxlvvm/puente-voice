import numpy as np
import pytest

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow, Say, State, Transfer
from puente_voice.safety import Alert
from puente_voice.safety_model import decide
from puente_voice.sim import demo_data

LABELS = ["any", "suicidal", "breathing", "chest_pain", "stroke", "overdose", "pregnancy_bleeding", "bleeding",
          "abuse", "emergency"]


class Stub:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def check(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def flow(model):
    audit = AuditLog()
    f = CallFlow("c", demo_data.CLINIC, demo_data.scheduler(), audit, now=lambda: demo_data.NOW, safety_model=model)
    f.start()
    return f, audit


def say(acts):
    return " ".join(a.text for a in acts if isinstance(a, Say))


def test_decide_fires_above_threshold_with_the_top_category():
    p = np.array([0.95, .1, .2, .1, .1, .7, .1, .1, .1, .1])
    assert decide(p, LABELS, 0.9) == Alert("overdose", "model")
    assert decide(np.array([0.5] + [0.9] * 9), LABELS, 0.9) is None


def test_model_only_alert_gets_the_quiet_review_transfer():
    f, audit = flow(Stub(Alert("chest_pain", "model")))
    acts = f.handle("Siento algo raro, como un peso aquí adelante")
    assert isinstance(acts[-1], Transfer) and acts[-1].reason == "safety_review:chest_pain"
    assert say(acts) == f.r.text("safety_review", "es") and "911" in say(acts)


def test_model_abuse_alert_gets_the_discreet_script():
    f, _ = flow(Stub(Alert("abuse", "model")))
    assert say(f.handle("Él está aquí, no puedo hablar mucho")) == f.r.text("emergency_abuse", "es")


def test_rules_decide_first_and_keep_the_emergency_script():
    model = Stub(Alert("chest_pain", "model"))
    f, _ = flow(model)
    acts = f.handle("No puedo respirar")
    assert acts[-1].reason == "emergency:breathing" and say(acts) == f.r.text("emergency", "es")


def test_a_failing_model_is_logged_and_the_call_goes_on():
    f, audit = flow(Stub(error=RuntimeError("onnx")))
    acts = f.handle("Español")
    assert not any(isinstance(a, Transfer) for a in acts) and f.state != State.TRANSFERRED
    assert any(e["kind"] == "safety_model_error" for e in audit.events)


def test_no_model_means_rules_only():
    f, _ = flow(None)
    assert not any(isinstance(a, Transfer) for a in f.handle("Español"))


def test_the_model_also_listens_after_the_goodbye():
    model = Stub(Alert("suicidal", "model"))
    f, _ = flow(model)
    f.state = State.DONE
    acts = f.handle("ya no quiero seguir así")
    assert acts[-1].reason == "safety_review:suicidal"
