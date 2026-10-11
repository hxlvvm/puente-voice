"""Rule-based bilingual emergency detector. Independent of the LLM; when in doubt it fires.

Patterns are written accent-free because input is folded first. A match is suppressed only if a negator stands right
before it (at most 2 bridge words between) in the same clause; kinds in ALWAYS_FIRE are never suppressed.
"""
import re
from dataclasses import dataclass

from puente_voice.text import fold

RULES = {
    "suicidal": [r"quitar(me|se)? la vida", r"suicid", r"matarme", r"no quiero (seguir )?vivir", r"hacerme dano",
                 r"kill myself", r"end my life", r"hurt myself", r"don'?t want to (live|be alive)",
                 r"\bme quiero morir", r"quiero morirme", r"(acabar|terminar) con (mi vida|todo\b(?! (el|la|los|las|esto|eso)\b))",
                 r"\bend(ing)? it all\b", r"take my (own )?life", r"better off dead", r"no vale la pena vivir",
                 r"\b(me|se) quier(o|e|en) (matar|morir)\b", r"\bmatarse\b", r"no quier(o|e|en) (seguir )?(vivi(r|endo)|estar viv[oa])",
                 r"(dormir|dormirme|dormirse) y no despertar", r"no (quiero|quiere) despertar(me)? (nunca|jamas)",
                 r"sentido a (seguir|vivir|la vida)", r"no (hubiera|haber) nacido", r"pegar(me|se) un tiro",
                 r"\bahorcar(me|se)\b", r"(aventar|tirar)(me|se)? (del|de un|de una|de la|desde el) (balcon|puente|edificio|azotea)",
                 r"cort(o|e|arse|arme) (las|sus|mis) munecas",
                 r"\bkill(ing|ed)? (my|him|her|them)sel(f|ves)\b", r"\b(want|wants|wanted) to die\b",
                 r"wish(es|ed)? (i|he|she|they) (was|were) dead", r"(thinking|thought|feel|feeling) (about|of|like) (ending it|dying)\b", r"sleep and (never|not) wake up",
                 r"don'?t want to be (here|alive|around) any ?more", r"better off without me",
                 r"\b(shoot|hang) (my|him|her|them)sel(f|ves)\b", r"jump(ing)? (off|from) (the|a) (bridge|building|roof)",
                 r"cut (my|his|her|their) wrists"],
    "breathing": [r"no puedo respirar", r"me falta el aire", r"falta de aire", r"me ahogo", r"dificultad para respirar",
                  r"can'?t breathe", r"can ?not breathe", r"short(ness)? of breath", r"trouble breathing",
                  r"difficulty breathing", r"hard to breathe", r"(se esta|esta|me estoy) ahogando",
                  r"no (me|le|les) entra (el )?aire", r"\bchoking\b", r"turning (blue|purple)",
                  r"(se esta poniendo|se puso) (morad[oa]|azul)", r"can'?t catch (my|his|her|their) breath",
                  r"no (puede|pueden) respirar", r"(he|she|they) can'?t breathe"],
    "chest_pain": [r"dolor (en el|de|del) pecho", r"me duele (mucho )?el pecho", r"presion en el pecho",
                   r"opresion en el pecho", r"chest pain", r"pain in (my|the) chest", r"chest (hurts|is hurting)",
                   r"pressure in (my|the) chest", r"tightness in (my|the) chest", r"infarto", r"heart attack",
                   r"ataque (al corazon|cardiaco)", r"(aprieta|apretando|presiona|oprime) el pecho",
                   r"dolor (\w+ ){0,3}en el pecho", r"pecho.{0,40}brazo izquierdo", r"brazo izquierdo.{0,40}pecho",
                   r"chest (is |feels )?(tight|heavy|squeez)", r"(pain|pressure).{0,30}(down|into) my (left )?arm"],
    "stroke": [r"no puedo mover (el|la|mi) (brazo|pierna|lado)", r"se me (cayo|torcio) la cara",
               r"no (puedo|puede) hablar bien(?! (el |en )?(ingles|espanol|english|spanish))", r"se me durmio (medio|la mitad del) (cuerpo|cara)", r"face (is )?droop",
               r"can'?t move my (arm|leg)", r"slurred speech", r"numb(ness)? on one side", r"\bstroke\b",
               r"derrame cerebral", r"embolia", r"se le (cayo|torcio|esta cayendo) la cara", r"(cara|boca) (chueca|torcida|caida)",
               r"habl(a|o|ando) (enredado|arrastrado)",
               r"no (siento|siente) (el|la|mi|su) (brazo|pierna|lado|cara)", r"(face|mouth) (is )?(drooping|droopy|crooked)",
               r"slurr", r"can'?t (lift|feel) (my|his|her) (arm|leg)", r"(he|she) can'?t move (his|her) (arm|leg)"],
    "overdose": [r"(tome|se tomo|se trago|trague) (todas las|muchas|varias) pastillas", r"sobredosis", r"overdose",
                 r"(took|swallowed) (all|too many|a lot of) (the |my |her |his )?(pills|tablets)",
                 r"(tomo|tome|trago|trague) (un|una|el|la|todo el|toda la) (frasco|bote|caja|botella)( enter[oa]| complet[oa])? de "
                 r"(pastillas|pildoras|medicinas?|cloro|veneno)",
                 r"(me tome|se tomo|se trago|trague) demasiad[ao]s (pastillas|pildoras|medicinas)", r"envenen",
                 r"(took|swallowed|ate) (way too many|a whole bottle of|the whole bottle of) (pills|tablets|medicine)"],
    "pregnancy_bleeding": [r"embarazada.{0,40}sangr", r"sangr.{0,40}embarazada", r"pregnant.{0,40}bleed",
                           r"bleed.{0,40}pregnan",
                           r"(encinta|esperando (un )?bebe|weeks along).{0,60}(sangr|blood|bleed)",
                           r"(sangr|blood|bleed).{0,60}(encinta|esperando (un )?bebe|weeks along)"],
    "bleeding": [r"sangrado (muy )?(fuerte|abundante)", r"estoy sangrando mucho", r"no (para|deja) de sangrar",
                 r"no paro de sangrar", r"bleeding (a lot|heavily|badly)", r"heavy bleeding", r"won'?t stop bleeding",
                 r"(sale|saliendo|salio|echando|botando) (much(isima|a)|bastante) sangre", r"sangr.{0,40}no (la|lo) puedo (parar|detener)", r"hemorragia",
                 r"blood (is )?(soaking|pouring|gushing|everywhere)", r"soak(ing|ed) through",
                 r"can'?t stop the bleeding"],
    "abuse": [r"me (pega|golpea|maltrata)", r"violencia (domestica|en (mi|la) casa)", r"abuso de mi",
              r"(he|she|they) (hits|beats|hurts) me", r"domestic violence", r"being abused", r"abused me",
              r"me (estan )?(maltratando|maltratan|amenaza|amenazan)",
              r"\b(he|she|they|someone|somebody|my (husband|wife|boyfriend|girlfriend|partner|father|mother|dad|mom|son|"
              r"daughter|brother|sister|ex)) (hit|beat|threatened|hits|beats|threatens) me\b",
              r"\b(he|she|they|someone|somebody)\b.{0,25}\b(is|are|'s|'re) (hurting|hitting) me\b"],
    "emergency": [r"\bemergencia\b", r"\bemergency\b", r"\bambulancia\b", r"\bambulance\b", r"\b911\b",
                  r"urgentisim", r"(mande|manden|envien|mandan) ayuda", r"send help"],
}
ALWAYS_FIRE = {"suicidal", "overdose", "abuse"}
# "never"/"nunca" are not here: "never had chest pain like this" is a symptom report
NEGATION = {"no", "sin", "tampoco", "ni", "not", "don't", "dont", "without", "doesn't", "isn't", "haven't", "hasn't",
            "didn't"}
BRIDGE = {"tengo", "tiene", "tenia", "hay", "have", "has", "had", "any", "a", "an", "un", "una", "el", "la", "ningun",
          "ninguna", "es", "is"}
CLAUSE_SPLIT = r"[.,;:!?]|\bpero\b|\bbut\b|\by\b|\band\b"
# a symptom placed clearly in the past ("el año pasado", "years ago") is history, unless something says it is back now
PAST = re.compile(r"\b(el|la) (ano|semana|mes|invierno|verano) pasad[oa]\b|\bhace (anos|meses|tiempo|mucho)\b|"
                  r"\b(years|months) ago\b|\blast (year|winter|summer|month)\b|\bwhen i was (a kid|young)\b|"
                  r"\bde nin[oa]\b")
NOW = re.compile(r"\b(ahora|ahorita|hoy|otra vez|de nuevo|todavia|now|again|today|tonight|still|back)\b")


@dataclass(frozen=True)
class Alert:
    kind: str
    source: str = "rules"   # "model": the safety classifier fired and the rules did not


class SafetyMonitor:
    def __init__(self, rules=RULES):
        self.rules = {k: [re.compile(p) for p in ps] for k, ps in rules.items()}

    def check(self, text: str):
        t = fold(text)
        for kind, patterns in self.rules.items():
            for p in patterns:
                for m in p.finditer(t):
                    if kind in ALWAYS_FIRE:
                        return Alert(kind)
                    if not self._negated(t, m) and not (PAST.search(self._clause(t, m)) and not NOW.search(t)):
                        return Alert(kind)
        return None

    @staticmethod
    def _clause(t, m):
        """The clause holding the match: a past phrase elsewhere in the turn must not silence a present symptom."""
        return re.split(CLAUSE_SPLIT, t[:m.start()])[-1] + m.group(0) + re.split(CLAUSE_SPLIT, t[m.end():])[0]

    @staticmethod
    def _negated(t, m):
        """Negator right before the match, allowing at most 2 bridge words ("no tengo ningún dolor de pecho")."""
        if m.group(0).split()[0] in NEGATION:  # the phrase itself is negative: "no puedo respirar"
            return False
        before = re.split(CLAUSE_SPLIT, t[:m.start()])[-1].split()
        i, hops = len(before) - 1, 0
        while i >= 0 and before[i] in BRIDGE and hops < 2:
            i, hops = i - 1, hops + 1
        return i >= 0 and before[i] in NEGATION
