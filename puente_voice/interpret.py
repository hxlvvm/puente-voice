"""Deterministic interpreter for caller text (es/en). The LLM interpreter in M3 has the same method names."""
import re
from datetime import date

from puente_voice.numbers import words_to_digits
from puente_voice.text import fold

ES_WORDS = {"hola", "si", "quiero", "necesito", "cita", "una", "buenos", "buenas", "llamo", "gracias", "por", "favor",
            "soy", "mi", "nombre", "tengo", "para", "el", "la", "de", "con", "que", "es", "espanol"}
EN_WORDS = {"hello", "hi", "yes", "english", "i", "need", "appointment", "my", "name", "is", "thank", "thanks", "the",
            "have", "want", "please", "an", "for", "with", "this"}
YES = r"\b(si|claro|correcto|exacto|esta bien|de acuerdo|ok|okay|perfecto|asi es|vale|eso es|andale|yes|yeah|yep|" \
      r"correct|right|sure|" \
      r"go ahead|please do|that's right)\b"
NO = r"\b(no|not|never|nunca|incorrecto|negativo|nope|wrong|incorrect)\b|n't\b"
NEGATED_YES = r"\b(not|no es|isn't|is not)\s+(right|correct|ok|okay|correcto|cierto)\b"
CONDITIONAL = r"\b(pero|but|wait|espera|espere|mejor)\b|\?"
HEDGE = r"\b(creo que|tal vez|quizas|a lo mejor|puede ser|no se|mas o menos|i think|maybe|perhaps|not sure|i guess|" \
        r"probably)\b"
PERSONA = r"\b(persona|representante|agente|operador|operadora|humano|alguien (de verdad|real)|" \
          r"representative|agent|operator|human|(an actual|a live|a real) person|someone real)\b|" \
          r"\b(comunic\w*|hablar|pase\w*|transfer\w*|talk|speak|connect)\b.{0,25}\b(recepcion|recepcionista|" \
          r"receptionist|front desk)\b"  # staff only as a request: "la recepcionista me dijo" is no transfer
FILLERS = {"hola", "buenos", "buenas", "dias", "tardes", "noches", "si", "yes", "ok", "okay", "claro", "bueno", "pues",
           "hello", "hi", "well", "sure", "good", "morning", "afternoon", "evening", "este", "mire", "ay", "eh", "o",
           "sea", "uh", "um", "oh", "yeah", "so"}
# courtesy words after the name: "Carmen Rivera, señorita", "Jennifer O'Connor calling", "..., para servirle"
TAIL = {"senorita", "senor", "senora", "para", "servirle", "gracias", "por", "favor", "calling", "speaking", "here",
        "thanks", "please"}
NAME_PREFIXES = [("mi", "nombre", "completo", "es"), ("mi", "nombre", "es"), ("me", "llamo"), ("yo", "soy"), ("soy",), ("es",), ("habla",),
                 ("my", "full", "name", "is"), ("my", "name", "is"), ("this", "is"), ("i", "am"), ("i'm",), ("it's",),
                 ("it", "is")]
NOT_NAMES = {"quiero", "cita", "necesito", "appointment", "need", "want", "can", "could", "you", "repeat", "sorry",
             "que", "como", "favor", "please", "hablar", "speak", "doctor", "si", "yes", "no", "una", "the",
             "gracias", "thank", "thanks", "video", "watching"}  # last line: Whisper's stock noise outputs
LANGS = {"ingles": "en", "english": "en", "espanol": "es", "spanish": "es"}
OTHER = {"en": "es", "es": "en"}
MONTHS_ES = {m: i + 1 for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                             "septiembre", "octubre", "noviembre", "diciembre"])}
MONTHS_EN = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                             "september", "october", "november", "december"])}
# ordinal words only; a number counts only as "opción/option/número N" or as the whole reply (times are numbers too)
ORDINALS = [(r"\b(primer[oa]?|first)\b", 0), (r"\b(segund[oa]|second)\b", 1), (r"\b(tercer[oa]?|third)\b", 2),
            (r"\b(cuart[oa]|fourth)\b", 3), (r"\b(quint[oa]|fifth)\b", 4)]
NUMBERS = {"1": 0, "uno": 0, "una": 0, "one": 0, "2": 1, "dos": 1, "two": 1, "3": 2, "tres": 2, "three": 2,
           "4": 3, "cuatro": 3, "four": 3, "5": 4, "cinco": 4, "five": 4}


def _valid(y, m, d):
    try:
        dt = date(y, m, d)
    except ValueError:
        return None
    return dt if date(1900, 1, 1) <= dt <= date.today() else None


def _year(s):
    """'1980' -> 1980; spoken two-digit years: '92' -> 1992, '05' -> 2005."""
    y = int(s)
    return y if len(s) == 4 else 1900 + y if y >= 30 else 2000 + y


class RuleInterpreter:
    def __init__(self, clinic):
        self.clinic = clinic

    def switch_request(self, text):
        """Only an explicit request: "no hablo inglés" -> es, "speak English" -> en, or just the language word."""
        t = fold(text)
        m = re.search(r"\b(?:no hablo|no entiendo|don't speak|do not speak|can't speak|cannot speak|"
                      r"don't understand)\s+(ingles|english|espanol|spanish)\b", t)
        if m:
            return OTHER[LANGS[m.group(1)]]
        m = re.search(r"\b(?:speak|hablar|hablo|habla|hable|in|en)\s+(ingles|english|espanol|spanish)\b", t)
        if m:
            return LANGS[m.group(1)]
        m = re.fullmatch(r"\W*(ingles|english|espanol|spanish)(?:\W+(?:please|por favor))?\W*", t)
        return LANGS[m.group(1)] if m else None

    def language(self, text, min_margin=1):
        explicit = self.switch_request(text)
        if explicit:
            return explicit
        words = set(re.findall(r"[a-z']+", fold(text)))
        es, en = len(words & ES_WORDS), len(words & EN_WORDS)
        if es - en >= min_margin:
            return "es"
        if en - es >= min_margin:
            return "en"
        return None

    def yes_no(self, text):
        t = re.sub(r"\bmejor no\b", " no ", re.sub(NEGATED_YES, " no ", fold(text)))  # "mejor no" = "better not"
        if not t.strip() or re.search(HEDGE, t) or re.search(CONDITIONAL, t):
            return "unclear"
        y, n = bool(re.search(YES, t)), bool(re.search(NO, t))
        return "yes" if y and not n else "no" if n and not y else "unclear"

    def persona(self, text):
        return bool(re.search(PERSONA, fold(text)))

    def name(self, text):
        toks = [w for w in (w.strip(".,!?¿¡") for w in (text or "").split()) if w]
        low = [fold(w) for w in toks]
        i = 0
        while i < len(low) and low[i] in FILLERS:
            i += 1
        for p in NAME_PREFIXES:
            if tuple(low[i:i + len(p)]) == p:
                i += len(p)
                break
        toks, low = toks[i:], low[i:]
        while low and low[-1] in TAIL:
            toks, low = toks[:-1], low[:-1]
        if not 2 <= len(toks) <= 6 or any(w in NOT_NAMES for w in low) or \
                not all(re.fullmatch(r"[A-Za-zÀ-ÿ'\-]+", w) for w in toks):
            return None
        return " ".join(toks)

    def dob(self, text, lang):
        t = words_to_digits(fold(text))
        m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", t)
        if m:
            a, b, y = map(int, m.groups())
            day, month = (a, b) if lang == "es" else (b, a)
            return _valid(y, month, day)
        m = re.search(r"\b(\d{1,2}) de (" + "|".join(MONTHS_ES) + r") (?:de |del )?(\d{4}|\d{2})\b", t)
        if m:
            return _valid(_year(m.group(3)), MONTHS_ES[m.group(2)], int(m.group(1)))
        m = re.search(r"\b(" + "|".join(MONTHS_EN) + r") (\d{1,2})(?:st|nd|rd|th)?,? (\d{4}|\d{2})\b", t)
        if m:
            return _valid(_year(m.group(3)), MONTHS_EN[m.group(1)], int(m.group(2)))
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)? of (" + "|".join(MONTHS_EN) + r"),? (\d{4}|\d{2})\b", t)
        if m:
            return _valid(_year(m.group(3)), MONTHS_EN[m.group(2)], int(m.group(1)))
        return None

    def visit(self, text):
        t = fold(text)
        hits = []
        for v in self.clinic.visit_types:
            words = [v.es, v.en, v.code.replace("-", " ")] + v.synonyms_es + v.synonyms_en
            if any(re.search(r"\b" + re.escape(fold(w)) + r"(e?s)?\b", t) for w in words):
                hits.append(v.code)
        return hits[0] if len(hits) == 1 else None

    def choice(self, text, n):
        t = re.sub(r"\b(un|a) (segundo|momento|second|moment|sec)\b", " ", fold(text))  # "wait a second" is no pick
        found = {i for pat, i in ORDINALS if re.search(pat, t)}
        if re.search(r"\b(ultim[oa]|last)\b", t):
            found.add(n - 1)
        if n % 2 and re.search(r"\b(medio|middle)\b", t):
            found.add(n // 2)
        for word in re.findall(r"\bla (uno|dos|tres|cuatro|cinco)\b", t):  # "la uno"; not "la una", a time
            found.add(NUMBERS[word])
        for word in re.findall(r"\b(?:opcion|option|numero|number)\s+(\w+)", t):
            if word in NUMBERS:
                found.add(NUMBERS[word])
        whole = re.fullmatch(r"\W*(\w+)\W*", t)
        if whole and whole.group(1) in NUMBERS:
            found.add(NUMBERS[whole.group(1)])
        return found.pop() if len(found) == 1 and next(iter(found)) < n else None

    def phone(self, text):
        digits = re.sub(r"\D", "", text or "")
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits if len(digits) == 10 else None
