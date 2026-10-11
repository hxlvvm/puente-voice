"""Deterministic call flow. The state machine decides every action; the interpreter only reads caller text."""
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from puente_voice.interpret import RuleInterpreter
from puente_voice.render import Renderer
from puente_voice.safety import SafetyMonitor
from puente_voice.scheduling.base import SchedulerError, SlotUnavailable


@dataclass
class Say:
    text: str
    lang: str


@dataclass
class Transfer:
    reason: str


@dataclass
class End:
    pass


class State(str, Enum):
    LANGUAGE = "language"
    NAME = "name"
    DOB = "dob"
    DOB_CONFIRM = "dob_confirm"
    NEW_PATIENT = "new_patient"
    PHONE = "phone"
    VISIT = "visit"
    OFFER = "offer"
    READBACK = "readback"
    DONE = "done"
    TRANSFERRED = "transferred"


class CallFlow:
    MAX_FAILS = 2

    def __init__(self, call_id, clinic, scheduler, audit, now, interpreter=None, safety=None, renderer=None,
                 safety_model=None):
        self.call_id, self.clinic, self.scheduler, self.audit, self.now = call_id, clinic, scheduler, audit, now
        self.ip = interpreter or RuleInterpreter(clinic)
        self.safety = safety or SafetyMonitor()
        self.safety_model = safety_model   # optional second layer, ORed with the rules
        self.r = renderer or Renderer(clinic)
        self.state, self.lang, self.fails = State.LANGUAGE, "es", 0
        self.name = self.dob = self.patient = self.visit = self.choice = self.booking = None
        self.phone = self.transfer_reason = None
        self.offered = []
        self.readback_repeats = self.reoffers = self.retaken = 0
        self.name_retried = False

    # ---- public -------------------------------------------------------------------------------
    def start(self):
        self.audit.event(self.call_id, "disclosure_played")
        return [Say(self.r.disclosure(), "bi")]

    def handle(self, utterance):
        if self.state == State.TRANSFERRED:
            return []
        text = (utterance or "").strip()
        if self.state == State.DONE:  # an emergency said after the goodbye still gets the emergency script
            alert = self._safety_check(text) if text else None
            if not alert:
                return []
            lang = self.ip.language(text) or self.lang
            return self._emergency(alert, lang)
        if text:
            alert = self._safety_check(text)
            if alert:
                lang = self.ip.language(text) or self.lang
                return self._emergency(alert, lang)
            if self.ip.persona(text):
                return self._transfer("caller_request", lang=self.ip.language(text) or self.lang)
            if self.state != State.LANGUAGE:
                switch = self.ip.switch_request(text)
                if switch and switch != self.lang:
                    self.lang = switch
                    self.audit.event(self.call_id, "language", lang=switch)
                    return [self._say(self._prompt())]
                if switch:  # asked for the language already in use: not an answer, counts as a failed turn
                    text = ""
        out = getattr(self, f"_on_{self.state.value}")(text) if text else None
        if out is None:
            self.fails += 1
            self.audit.event(self.call_id, "turn_failed", state=self.state.value, count=self.fails)
            if self.fails >= self.MAX_FAILS:
                return self._transfer("not_understood")
            return [self._say(self.r.text("repeat", self.lang))]
        self.fails = 0
        return out

    # ---- helpers ------------------------------------------------------------------------------
    def _say(self, text):
        return Say(text, self.lang)

    def _emergency(self, alert, lang):
        self.audit.event(self.call_id, "emergency", reason=alert.kind)
        if alert.kind == "abuse":
            script = "emergency_abuse"
        else:
            script = "emergency" if alert.source == "rules" else "safety_review"
        prefix = "emergency" if alert.source == "rules" else "safety_review"
        return self._transfer(f"{prefix}:{alert.kind}", self.r.text(script, lang), lang)

    def _safety_check(self, text):
        """Rules first; the model can only add an alert. A failing model is logged and the rules decide."""
        alert = self.safety.check(text)
        if alert or self.safety_model is None:
            return alert
        try:
            return self.safety_model.check(text)
        except Exception:  # never let the second layer break or silence the call
            self.audit.event(self.call_id, "safety_model_error")   # audit fields are allowlisted; no text
            return None

    def _transfer(self, reason, text=None, lang=None):
        self.state, self.transfer_reason = State.TRANSFERRED, reason
        self.audit.event(self.call_id, "transfer", reason=reason)
        lang = lang or self.lang
        return [Say(text or self.r.text("transfer", lang), lang), Transfer(reason)]

    def _prompt(self):
        s, L = self.state, self.lang
        if s == State.NAME:
            return self.r.text("ask_name", L)
        if s == State.DOB:
            return self.r.text("ask_dob", L)
        if s == State.DOB_CONFIRM:
            return self.r.text("confirm_dob", L, dob=self.r.date(self.dob, L))
        if s == State.NEW_PATIENT:
            return self.r.text("ask_new", L)
        if s == State.PHONE:
            return self.r.text("ask_phone", L)
        if s == State.VISIT:
            return self.r.text("ask_visit", L, options=self.r.visit_options(L))
        if s == State.OFFER:
            return self.r.offer(self.offered, L)
        if s == State.READBACK:
            return self.r.readback(self.choice, self.visit, L)
        return self.r.text("repeat", L)

    def _go(self, state):
        self.state = state
        return [self._say(self._prompt())]

    # ---- states -------------------------------------------------------------------------------
    def _on_language(self, text):
        lang = self.ip.language(text)
        if not lang:
            return None
        self.lang = lang
        self.audit.event(self.call_id, "language", lang=lang)
        return self._go(State.NAME)

    def _on_name(self, text):
        self.name = self.ip.name(text)
        if not self.name:
            return None
        return self._lookup() if self.name_retried else self._go(State.DOB)

    def _on_dob(self, text):
        self.dob = self.ip.dob(text, self.lang)
        return self._go(State.DOB_CONFIRM) if self.dob else None

    def _on_dob_confirm(self, text):
        yn = self.ip.yes_no(text)
        if yn == "no":
            self.dob = None
            return self._go(State.DOB)
        if yn != "yes":
            return None
        return self._lookup()

    def _lookup(self):
        try:
            matches = self.scheduler.find_patients(self.name, self.dob)
        except SchedulerError:
            return self._transfer("scheduler_error")
        if len(matches) == 1:
            self.patient = matches[0]
            self.audit.event(self.call_id, "identity_matched", patient_id=self.patient.id)
            return self._go(State.VISIT)
        if not matches and not self.name_retried:  # a misheard name is likelier than a new patient: ask once more
            self.name_retried = True
            self.state = State.NAME
            return [self._say(self.r.text("ask_name_again", self.lang))]
        if not matches:
            return self._go(State.NEW_PATIENT)
        return self._transfer("ambiguous_match")

    def _on_new_patient(self, text):
        yn = self.ip.yes_no(text)
        if yn == "yes":
            return self._go(State.PHONE)
        if yn == "no":
            return self._transfer("no_match")
        return None

    def _on_phone(self, text):
        self.phone = self.ip.phone(text)
        return self._transfer("new_patient") if self.phone else None

    def _on_visit(self, text):
        self.visit = self.ip.visit(text)
        return self._offer() if self.visit else None

    def _offer(self):
        start = self.now()
        try:
            slots = self.scheduler.search_slots(self.visit, start, start + timedelta(days=self.clinic.search_days),
                                                self.clinic.max_slots_offered)
        except SchedulerError:
            return self._transfer("scheduler_error")
        if not slots:
            return [self._say(self.r.text("no_slots", self.lang))] + self._transfer("no_slots")
        self.offered = slots
        return self._go(State.OFFER)

    def _on_offer(self, text):
        i = self.ip.choice(text, len(self.offered))
        if i is None:
            return None
        self.choice, self.readback_repeats = self.offered[i], 0
        return self._go(State.READBACK)

    def _on_readback(self, text):
        yn = self.ip.yes_no(text)
        if yn == "yes":
            return self._book()
        if yn == "no":
            remaining = [s for s in self.offered if s != self.choice]
            if self.reoffers < 1 and remaining:
                self.reoffers += 1
                self.offered = remaining
                return self._go(State.OFFER)
            return self._transfer("declined")
        if self.readback_repeats < 1:
            self.readback_repeats += 1
            return [self._say(self.r.text("confirm_yes_no", self.lang))]
        return self._transfer("unclear_confirmation")

    def _book(self):
        try:
            self.booking = self.scheduler.book(self.choice.id, self.patient.id, idempotency_key=self.call_id)
        except SlotUnavailable:
            self.audit.event(self.call_id, "slot_unavailable", slot_id=self.choice.id)
            if self.retaken < 1:
                self.retaken += 1
                return [self._say(self.r.text("slot_taken", self.lang))] + self._offer()
            return self._transfer("slot_unavailable")
        except SchedulerError:
            return self._transfer("scheduler_error")
        self.audit.event(self.call_id, "booked", appointment_id=self.booking.appointment_id, slot_id=self.choice.id)
        self.state = State.DONE
        return [self._say(self.r.booked(self.choice, self.lang)), End()]
