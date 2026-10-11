from datetime import date, datetime, timedelta, timezone

import pytest

from puente_voice.scheduling.base import Patient, Slot, SlotUnavailable, names_match
from puente_voice.scheduling.memory import InMemoryScheduler

T0 = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)


def slots():
    return [Slot(f"s{i}", T0 + timedelta(hours=i), T0 + timedelta(hours=i, minutes=30), "general", "Dra. Ana García")
            for i in range(4)] + [Slot("v1", T0, T0 + timedelta(minutes=15), "vaccine", "")]


def sched():
    return InMemoryScheduler([Patient("p1", "María José Pérez García", date(1980, 3, 15)),
                              Patient("p2", "Ana López", date(1990, 1, 1)),
                              Patient("p3", "Ana López", date(1990, 1, 1))], slots())


def test_names_match_accents_and_compound():
    assert names_match("Maria Perez", "María José Pérez García")
    assert names_match("maría josé pérez garcía", "María José Pérez García")
    assert not names_match("Maria Lopez", "María Pérez")
    assert not names_match("Maria", "María Pérez")


def test_find_patients_by_name_and_dob():
    s = sched()
    assert [p.id for p in s.find_patients("Maria Perez", date(1980, 3, 15))] == ["p1"]
    assert s.find_patients("Maria Perez", date(1981, 3, 15)) == []
    assert len(s.find_patients("Ana Lopez", date(1990, 1, 1))) == 2


def test_search_slots_filters_type_window_and_limit():
    s = sched()
    got = s.search_slots("general", T0 + timedelta(minutes=1), T0 + timedelta(days=1), 2)
    assert [x.id for x in got] == ["s1", "s2"]


def test_book_is_idempotent_and_slot_cannot_be_double_booked():
    s = sched()
    b1 = s.book("s1", "p1", "call-1")
    assert s.book("s1", "p1", "call-1") == b1
    with pytest.raises(SlotUnavailable):
        s.book("s1", "p2", "call-2")
    s.take("s2")
    with pytest.raises(SlotUnavailable):
        s.book("s2", "p1", "call-3")
    assert "s1" not in [x.id for x in s.search_slots("general", T0, T0 + timedelta(days=1), 5)]
