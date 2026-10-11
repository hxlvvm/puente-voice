"""Hand-written training seeds for the suicidal category, which the local generator keeps softening into vague worry.

Written 2026-10-11 before looking at any blind-set suicidal item. Training data only, never evaluation. Each core
statement is wrapped in call contexts; calibration takes whole cores (hash split), like the generated families.
The same wrappers also carry ordinary clinic statements as negatives, so the wrapper itself is never a cue.

usage: python train/safety_seeds.py GEN_DIR OUT_DIR   (copies GEN_DIR's files and appends the seeds)
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

CORES = {
    "es": ["me quiero morir", "ya no quiero vivir", "estoy pensando en quitarme la vida", "voy a matarme",
           "tengo las pastillas guardadas para acabar con todo", "ya no quiero despertar mañana",
           "sería mejor para todos si yo no estuviera", "ya me despedí de mis hijos, no aguanto más",
           "no le veo sentido a nada, ya tengo pensado cómo hacerlo", "he estado pensando en hacerme daño",
           "me corté otra vez y quiero seguir", "esta noche me voy a tomar todo el frasco para no despertar",
           "mi hijo dice que se quiere matar", "mi hija escribió una carta de despedida y no la encuentro",
           "mi esposo dijo que se va a pegar un tiro", "mi mamá dice que ya no quiere seguir viviendo",
           "a veces pienso que lo mejor es desaparecer para siempre", "ya no puedo más con esta vida, me quiero ir",
           "estoy en el puente y pienso tirarme", "compré una cuerda, ya lo decidí"],
    "en": ["I want to die", "I don't want to be alive anymore", "I'm thinking about killing myself",
           "I'm going to end my life tonight", "I have pills saved up to end it", "I don't want to wake up tomorrow",
           "everyone would be better off without me", "I already said goodbye to my kids, I can't do this anymore",
           "there's no point, I've thought about how I'd do it", "I've been thinking about hurting myself",
           "I cut myself again and I want to keep going", "my son says he wants to kill himself",
           "my daughter left a goodbye note and I can't find her", "my husband said he's going to shoot himself",
           "my mom keeps saying she wants to die", "I'm on the bridge and I'm thinking of jumping",
           "I bought a rope, I've made up my mind", "I just want it all to stop for good"],
}
NEUTRAL = {
    "es": ["necesito cambiar mi cita del martes", "perdí mi tarjeta del seguro", "mi hijo tiene tos desde hace una semana",
           "no sé si mi seguro cubre la consulta", "se me acabaron las pastillas de la presión",
           "no entendí lo que me dijo el doctor", "llegué tarde a la cita de ayer", "mi mamá necesita un chequeo",
           "no tengo quién me lleve a la clínica", "quiero saber los resultados de mis análisis",
           "me duele la rodilla cuando camino", "ya no me alcanza para pagar la consulta"],
    "en": ["I need to move my Tuesday appointment", "I lost my insurance card", "my son has had a cough for a week",
           "I'm not sure my insurance covers the visit", "I ran out of my blood pressure pills",
           "I didn't understand what the doctor told me", "I missed my appointment yesterday",
           "my mom needs a checkup", "I don't have a ride to the clinic", "I want to know my lab results",
           "my knee hurts when I walk", "I can't afford the visit anymore"],
}
WRAP = {
    "es": ["{c}.", "Mire, {c}.", "Sí, quiero la cita, pero la verdad... {c}.", "Este... no sé cómo decirlo, {c}.",
           "Perdone, antes de la cita le quiero decir que {c}.", "Ay, {c}, ya no sé qué hacer."],
    "en": ["{c}.", "Honestly, {c}.", "Yeah, I need the appointment, but... {c}.", "Um, I don't know how to say this, {c}.",
           "Before we book anything, {c}.", "I'm sorry, {c}, I don't know what to do."],
}


def rows():
    out = []
    for kind, cat, table in (("emergency", "suicidal", CORES), ("ordinary", None, NEUTRAL)):
        for lang, cores in table.items():
            for i, core in enumerate(cores):
                fam = f"seed/{cat or kind}/{lang}/{i}"
                split = "calib" if int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 100 < 15 else "train"
                for w in WRAP[lang]:
                    text = w.format(c=core)
                    out.append({"family": fam, "kind": kind, "category": cat, "lang": lang,
                                "text": text[0].upper() + text[1:], "split": split})
    return out


def main(gen_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = rows()
    for split in ("train", "calib"):
        shutil.copy(Path(gen_dir) / f"gen_{split}.jsonl", out / f"gen_{split}.jsonl")
        with open(out / f"gen_{split}.jsonl", "a", encoding="utf-8") as fh:
            for r in seeds:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({s: sum(r["split"] == s for r in seeds) for s in ("train", "calib")}, "seed turns added")


if __name__ == "__main__":
    main(*sys.argv[1:])
