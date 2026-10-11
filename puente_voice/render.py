"""Fixed es/en templates. Everything the agent says about dates, times, slots and bookings comes from here."""
from datetime import date

DAYS = {"es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
MONTHS = {"es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                 "noviembre", "diciembre"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]}
ORDINALS = {"es": ["primera opción", "segunda opción", "tercera opción", "cuarta opción", "quinta opción"],
            "en": ["first", "second", "third", "fourth", "fifth"]}

TEMPLATES = {
    "ask_name": {"es": "Por favor, dígame su nombre completo.", "en": "Please tell me your full name."},
    "ask_name_again": {"es": "No encuentro su expediente con ese nombre. ¿Me puede repetir su nombre completo, por favor?",
                       "en": "I couldn't find a record with that name. Could you tell me your full name again, please?"},
    "ask_dob": {"es": "Gracias. ¿Cuál es su fecha de nacimiento?", "en": "Thank you. What is your date of birth?"},
    "confirm_dob": {"es": "Tengo su fecha de nacimiento como {dob}. ¿Es correcto?",
                    "en": "I have your date of birth as {dob}. Is that correct?"},
    "ask_new": {"es": "No encuentro su expediente. ¿Es usted paciente nuevo?",
                "en": "I couldn't find your record. Are you a new patient?"},
    "ask_phone": {"es": "¿A qué número de teléfono le podemos llamar?", "en": "What phone number can we reach you at?"},
    "ask_visit": {"es": "¿Qué tipo de cita necesita? Puede ser: {options}.",
                  "en": "What kind of appointment do you need? It can be: {options}."},
    "offer": {"es": "Tengo estas opciones: {options}. ¿Cuál prefiere?",
              "en": "I have these options: {options}. Which one would you like?"},
    "readback": {"es": "Su cita sería {slot}, para {visit}. ¿La reservo?",
                 "en": "Your appointment would be {slot}, for a {visit}. Shall I book it?"},
    "booked": {"es": "Listo. Su cita quedó reservada {slot}. Gracias por llamar. Hasta luego.",
               "en": "Done. Your appointment is booked for {slot}. Thank you for calling. Goodbye."},
    "slot_taken": {"es": "Lo siento, ese horario acaba de ocuparse.", "en": "Sorry, that time was just taken."},
    "repeat": {"es": "Disculpe, no le entendí. ¿Me lo puede repetir?",
               "en": "Sorry, I didn't catch that. Could you say it again?"},
    "confirm_yes_no": {"es": "Por favor, responda sí o no.", "en": "Please answer yes or no."},
    "transfer": {"es": "Le comunico con una persona del equipo. Un momento, por favor.",
                 "en": "I'm transferring you to a member of our team. One moment, please."},
    # abuse: no loud emergency message, the abuser may be listening (owner-approved 2026-10-11, clinician review pending)
    "emergency_abuse": {"es": "Gracias por decírmelo. Le comunico ahora con una persona del equipo. Si está en peligro, "
                              "llame al 911 cuando pueda hacerlo con seguridad.",
                        "en": "Thank you for telling me. I'm connecting you with a member of our team now. If you are "
                              "in danger, call 911 when it is safe to do so."},
    # the safety model fired and the rules did not: a quieter transfer that still names 911 (clinician review pending)
    "safety_review": {"es": "Para ayudarle mejor, le comunico ahora con una persona del equipo. Si es una emergencia, "
                            "cuelgue y llame al 911.",
                      "en": "To help you better, I'm connecting you with a member of our team now. If this is an "
                            "emergency, hang up and call 911."},
    "emergency": {"es": "Si esto es una emergencia, cuelgue y llame al 911. Si está pensando en hacerse daño, llame o "
                        "envíe un mensaje al 988. Le comunico ahora con una persona del equipo.",
                  "en": "If this is an emergency, hang up and call 911. If you are thinking about hurting yourself, "
                        "call or text 988. I'm connecting you with a member of our team now."},
    "no_slots": {"es": "Lo siento, no tengo horarios disponibles para ese tipo de cita.",
                 "en": "Sorry, I don't have any open times for that kind of appointment."},
}


def _join(items, lang):
    if len(items) == 1:
        return items[0]
    if lang == "es":
        return ", ".join(items[:-1]) + " o " + items[-1]
    return ", ".join(items[:-1]) + ", or " + items[-1]


class Renderer:
    def __init__(self, clinic):
        self.clinic = clinic

    def text(self, key, lang, **kw):
        return TEMPLATES[key][lang].format(**kw)

    def disclosure_parts(self):
        """The disclosure as (text, language) parts, so each part is spoken with its own language's voice."""
        n = self.clinic.name
        # 911 first: the caller is not heard until the notice ends (owner decision 2026-10-12, clinician review pending)
        return [("Si es una emergencia o está en peligro, cuelgue y llame al 911.", "es"),
                ("If this is an emergency or you are in danger, hang up and call 911.", "en"),
                (f"Gracias por llamar a {n}. Esta llamada es atendida por un asistente automático, solo para citas, "
                 f"y se transcribe. Para hablar con una persona, diga «persona» en cualquier momento.", "es"),
                (f"Thank you for calling {n}. This call is answered by an automated assistant, for scheduling only, "
                 f"and is transcribed. To speak with a person, say 'representative' at any time.", "en"),
                ("¿En qué idioma prefiere continuar?", "es"), ("Which language would you prefer?", "en")]

    def disclosure(self):
        return " ".join(text for text, _ in self.disclosure_parts())

    def date(self, d: date, lang):
        if lang == "es":
            return f"{d.day} de {MONTHS['es'][d.month - 1]} de {d.year}"
        return f"{MONTHS['en'][d.month - 1]} {d.day}, {d.year}"

    def _time(self, t, lang):
        h12 = t.hour % 12 or 12
        hm = f"{h12}:{t.minute:02d}"
        if lang == "en":
            return f"{hm} {'AM' if t.hour < 12 else 'PM'}"
        part = ("de la mañana" if t.hour < 12 else "del mediodía" if t.hour == 12
                else "de la tarde" if t.hour < 19 else "de la noche")
        return f"{'a la' if h12 == 1 else 'a las'} {hm} {part}"

    def slot(self, slot, lang):
        t = slot.start.astimezone(self.clinic.tz)
        day = DAYS[lang][t.weekday()]
        month = MONTHS[lang][t.month - 1]
        if lang == "es":
            s = f"el {day} {t.day} de {month} {self._time(t, 'es')}"
            return s + (f" con {slot.provider}" if slot.provider else "")
        s = f"{day}, {month} {t.day} at {self._time(t, 'en')}"
        return s + (f" with {slot.provider}" if slot.provider else "")

    def offer(self, slots, lang):
        opts = [f"{ORDINALS[lang][i]}, {self.slot(s, lang)}" for i, s in enumerate(slots)]
        return self.text("offer", lang, options="; ".join(opts))

    def visit_options(self, lang):
        return _join([getattr(v, lang) for v in self.clinic.visit_types], lang)

    def readback(self, slot, visit_code, lang):
        return self.text("readback", lang, slot=self.slot(slot, lang), visit=getattr(self.clinic.visit(visit_code), lang))

    def booked(self, slot, lang):
        return self.text("booked", lang, slot=self.slot(slot, lang))
