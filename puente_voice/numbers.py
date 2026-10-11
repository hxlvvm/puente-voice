"""Spoken numbers to digits, so dates said in words parse like dates written in digits.

"quince de marzo de mil novecientos ochenta" -> "15 de marzo de 1980"; "May second, nineteen eighty five" ->
"May 2, 1985". Used only on date-of-birth answers; everything that is not a run of number words is left as it was.
"""
import re

from puente_voice.text import fold

_ES = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintiun": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100, "doscientos": 200,
    "trescientos": 300, "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600, "setecientos": 700,
    "ochocientos": 800, "novecientos": 900,
}
_EN = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_ORDINAL = {
    "primero": 1, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30,
}
WORDS = {**_ES, **_EN, **_ORDINAL}
_THOUSAND = {"mil", "thousand"}
_JOIN = {"y", "and"}
_NOT_ALONE = {"oh", "uno", "one"}  # alone these are interjections or pronouns more often than numbers


def _is_num(tok):
    f = fold(tok)
    return f in WORDS or f in _THOUSAND


def _value(words):
    """Integer for a run of number words; 'nineteen eighty five' style years become 1985."""
    total = current = 0
    first_pair = None
    for w in words:
        if w in _THOUSAND:
            total += (current or 1) * 1000
            current = 0
            continue
        if w == "hundred":
            current = (current or 1) * 100
            continue
        v = WORDS[w]
        if total == 0 and first_pair is None and current >= 10 and (w == "oh" or v >= 10):
            first_pair, current = current, v   # "nineteen" | "eighty five", "nineteen" | "oh five"
            continue
        current += v
    return first_pair * 100 + current if first_pair is not None else total + current


def words_to_digits(text):
    toks = re.findall(r"[^\W\d_]+|\d+|\s+|.", text or "")
    out, i, n = [], 0, len(toks)
    while i < n:
        if not _is_num(toks[i]):
            out.append(toks[i])
            i += 1
            continue
        words, end = [fold(toks[i])], i + 1
        while words[-1] not in _ORDINAL:          # an ordinal ends the number: "the fourth | of July"
            k = end
            if k < n and (toks[k].isspace() or toks[k] == "-"):
                k += 1
            else:
                break
            if k < n and fold(toks[k]) in _JOIN:     # "treinta y uno", "two hundred and five"
                k += 1
                if k < n and toks[k].isspace():
                    k += 1
            if k < n and _is_num(toks[k]):
                words.append(fold(toks[k]))
                end = k + 1
            else:
                break
        if len(words) == 1 and words[0] in _NOT_ALONE:
            out.append(toks[i])
            i += 1
            continue
        out.append(str(_value(words)))
        i = end
    return "".join(out)
