"""Pick the test caller voice whose recognition matches real recordings, never by the gate's own words.

  python eval/voice_match.py MODELS_DIR    (GPU; real clips from eval/fetch_eval + extract_eval)
"""
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

from puente_voice.evaluate import EXCLUDED_WORDS, choose_voice, phone_segment
from puente_voice.text import fold
from puente_voice.voice.speech import load_kokoro, synth

WORDS = ["hola", "claro", "vale", "bueno", "gracias", "correcto", "primera", "segunda"]
SPANISH_VOICES = ["ef_dora", "em_alex", "em_santa"]
SPEEDS = (0.9, 1.0, 1.1)  # Kokoro is deterministic: vary the rate to get three takes per word


def norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", fold(t)))


def main(models):
    from faster_whisper import WhisperModel
    m = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")

    def heard(x):
        segs, _ = m.transcribe(x, language="es", vad_filter=True)
        return norm(" ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.4))

    real = {}
    for w in WORDS:
        clips = sorted(glob.glob(f"{models}/eval/words/{w}_common_voice_*.wav"))[:25]
        ok = 0
        for p in clips:
            x, sr = sf.read(p, dtype="float32")
            ok += heard(phone_segment((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(), sr)) == w
        real[w] = ok / len(clips)
    k = load_kokoro(f"{models}/kokoro", int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    voices = {}
    for v in SPANISH_VOICES:
        acc = {}
        for w in WORDS:
            ok = 0
            for speed in SPEEDS:
                wav, sr = k.create(w, voice=v, lang="es", speed=speed)
                ok += heard(phone_segment((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes(), sr)) == w
            acc[w] = ok / len(SPEEDS)
        voices[v] = acc
    print("word       real  " + "  ".join(f"{v:>8}" for v in voices))
    for w in WORDS:
        print(f"{w:9} {real[w]:5.2f}  " + "  ".join(f"{voices[v][w]:8.2f}" for v in voices))
    print(f"excluded from the choice: {sorted(EXCLUDED_WORDS)}")
    print(f"chosen caller voice: {choose_voice(real, voices)}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
