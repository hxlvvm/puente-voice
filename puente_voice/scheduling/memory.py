"""In-memory scheduler for tests and the simulator."""
from puente_voice.scheduling.base import Booking, SlotUnavailable, names_match


class InMemoryScheduler:
    def __init__(self, patients, slots):
        self.patients = list(patients)
        self.slots = {s.id: s for s in slots}
        self.busy: set[str] = set()
        self.appointments: dict[str, Booking] = {}
        self._by_key: dict[str, Booking] = {}

    def find_patients(self, name, birth_date):
        return [p for p in self.patients if p.birth_date == birth_date and names_match(name, p.name)]

    def search_slots(self, visit_type, start, end, limit):
        free = [s for s in self.slots.values()
                if s.visit_type == visit_type and start <= s.start < end and s.id not in self.busy]
        return sorted(free, key=lambda s: s.start)[:limit]

    def book(self, slot_id, patient_id, idempotency_key):
        if idempotency_key in self._by_key:
            return self._by_key[idempotency_key]
        if slot_id in self.busy or slot_id not in self.slots:
            raise SlotUnavailable(slot_id)
        self.busy.add(slot_id)
        b = Booking(f"a{len(self.appointments) + 1}", slot_id, patient_id)
        self.appointments[b.appointment_id] = b
        self._by_key[idempotency_key] = b
        return b

    def take(self, slot_id):
        """Simulate another caller booking the slot first."""
        self.busy.add(slot_id)
