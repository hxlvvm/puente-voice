"""Text-only simulated calls: scripted caller turns against CallFlow with the demo scheduler.

  python -m puente_voice.sim.text_sim scenarios
"""
import sys
from pathlib import Path

import yaml

from puente_voice.audit import AuditLog
from puente_voice.flow import CallFlow, Say
from puente_voice.sim import demo_data


def load_scenarios(directory):
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.yaml"))]


def check(spec, flow, said):
    """Problems with a finished call against the scenario's `expect` block; empty means it passed."""
    exp, problems = spec.get("expect", {}), []
    if "state" in exp and flow.state.value != exp["state"]:
        problems.append(f"state: expected {exp['state']}, got {flow.state.value}")
    booked = flow.booking.slot_id if flow.booking else None
    if "booked_slot" in exp and booked != exp["booked_slot"]:
        problems.append(f"booked_slot: expected {exp['booked_slot']}, got {booked}")
    if "transfer_reason" in exp:
        want, got = exp["transfer_reason"], flow.transfer_reason or ""
        if not (got.startswith(want[:-1]) if want.endswith("*") else got == want):
            problems.append(f"transfer_reason: expected {want}, got {got}")
    for s in exp.get("says_contains", []):
        if s not in said:
            problems.append(f"says_contains: missing {s!r}")
    return problems


def run_scenario(spec, safety_model=None):
    sched = demo_data.scheduler()
    flow = CallFlow(f"sim-{spec['name']}", demo_data.CLINIC, sched, AuditLog(), now=lambda: demo_data.NOW,
                    safety_model=safety_model)
    transcript = [("agent", a.text) for a in flow.start() if isinstance(a, Say)]
    take = spec.get("take_slot")
    for i, utt in enumerate(spec["caller"]):
        if take and take["before_turn"] == i:
            sched.take(take["slot"])
        transcript.append(("caller", utt))
        transcript += [("agent", a.text) for a in flow.handle(utt) if isinstance(a, Say)]
    booked = flow.booking.slot_id if flow.booking else None
    problems = check(spec, flow, " ".join(t for who, t in transcript if who == "agent"))
    return {"name": spec["name"], "passed": not problems, "problems": problems, "transcript": transcript,
            "state": flow.state.value, "booked_slot": booked, "transfer_reason": flow.transfer_reason}


def main(argv=None):
    args = argv or sys.argv[1:] or ["scenarios"]
    model = None
    if len(args) > 1:   # optional second argument: a trained safety-model directory (train/safety_train.py)
        from puente_voice.safety_model import SafetyModel
        model = SafetyModel(args[1])
    results = [run_scenario(s, model) for s in load_scenarios(args[0])]
    for r in results:
        outcome = r["booked_slot"] or r["transfer_reason"] or r["state"]
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']:<24} {outcome}  {'; '.join(r['problems'])}")
    failed = sum(not r["passed"] for r in results)
    print(f"{len(results) - failed}/{len(results)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
