import pytest

from puente_voice.numbers import words_to_digits


@pytest.mark.parametrize("text,want", [
    ("quince de marzo de mil novecientos ochenta", "15 de marzo de 1980"),
    ("el veintiuno de julio de mil novecientos setenta y cinco", "el 21 de julio de 1975"),
    ("treinta y uno de diciembre de dos mil cinco", "31 de diciembre de 2005"),
    ("primero de enero de mil novecientos noventa", "1 de enero de 1990"),
    ("May second, nineteen eighty five", "May 2, 1985"),
    ("the fourth of July nineteen seventy five", "the 4 of July 1975"),
    ("June twenty-first, two thousand three", "June 21, 2003"),
    ("March third nineteen oh five", "March 3 1905"),
    ("15 de marzo de 1980", "15 de marzo de 1980"),          # digits untouched
    ("una cita para el martes", "una cita para el martes"),    # 'una' alone is not a number in a sentence
])
def test_words_to_digits(text, want):
    assert words_to_digits(text) == want
