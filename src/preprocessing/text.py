"""Text cleaning, tokenisation and (light) lemmatisation for research text."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:  # spaCy is optional
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser"])
    HAS_SPACY = True
except Exception:  # pragma: no cover - depends on environment
    _NLP, HAS_SPACY = None, False

# Generic academic filler that carries no topical signal
ACADEMIC_STOP = {
    "propose", "present", "study", "paper", "approach", "using", "use", "used", "based", "evaluate",
    "compare", "novel", "method", "methods", "show", "shows", "we", "our", "analyse", "analyze",
    "design", "build", "builds", "results", "result", "work", "model", "models", "system", "systems",
    "trained", "train", "examine", "examines", "characterise", "large-scale",
}
STOP = set(ENGLISH_STOP_WORDS) | ACADEMIC_STOP

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")
_KEEP_S = ("ss", "us", "is", "ics", "sis")   # analysis, bias, graphics ... keep trailing s


def light_lemma(tok: str) -> str:
    """Tiny rule-based plural normaliser used when spaCy is not installed."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(_KEEP_S):
        return tok[:-1]
    return tok


def clean_text(text) -> str:
    """Lowercase + keep letters/digits/hyphens. Does not remove stop-words."""
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\-\s]", " ", text)).strip()


def preprocess(text, use_spacy: bool = None) -> str:
    """Full normalisation used by the lexical models (TF-IDF / BM25 / topics)."""
    text = clean_text(text)
    use = HAS_SPACY if use_spacy is None else (use_spacy and HAS_SPACY)
    if use:
        toks = [t.lemma_.lower() for t in _NLP(text) if not t.is_space]
    else:
        toks = [light_lemma(t) for t in _TOKEN.findall(text.lower())]
    return " ".join(t for t in toks if t not in STOP and len(t) > 1)


def tokens(text) -> list:
    return preprocess(text).split()
