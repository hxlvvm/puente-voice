from pathlib import Path

import pytest

from puente_voice.sim.text_sim import load_scenarios, run_scenario

SCEN = load_scenarios(Path(__file__).resolve().parent.parent / "scenarios")


def test_there_are_scenarios():
    assert len(SCEN) >= 14


@pytest.mark.parametrize("spec", SCEN, ids=[s["name"] for s in SCEN])
def test_scenario(spec):
    r = run_scenario(spec)
    assert r["passed"], r["problems"]


def test_runner_reports_failures():
    r = run_scenario({"name": "bad", "caller": ["Hola"], "expect": {"state": "done"}})
    assert not r["passed"] and "state" in r["problems"][0]
