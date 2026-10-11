import re
import unicodedata


def fold(s: str) -> str:
    """Lower-case, accents removed (ñ -> n), whitespace collapsed. Keeps apostrophes and punctuation."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()
