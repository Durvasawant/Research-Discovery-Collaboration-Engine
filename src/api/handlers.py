"""Framework-independent request handlers. Both the FastAPI app and the stdlib server call these, so behaviour is
identical and unit-testable. Every handler takes/returns plain JSON-serialisable dicts."""
from typing import Optional
from src.models.engine import ResearchDiscoveryEngine, METHODS

_ENGINE: Optional[ResearchDiscoveryEngine] = None


def get_engine() -> ResearchDiscoveryEngine:
    global _ENGINE
    if _ENGINE is None:
        from src.models.engine import load_engine
        _ENGINE = load_engine()
    return _ENGINE


def set_engine(engine: ResearchDiscoveryEngine):
    global _ENGINE
    _ENGINE = engine


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message); self.status, self.message = status, message


def _int(v, name, lo, hi, default):
    v = default if v is None else v
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ApiError(422, f"'{name}' must be an integer in [{lo}, {hi}]")
    return v


def _text(body, key="text", max_len=20000):
    t = body.get(key)
    if not isinstance(t, str) or not t.strip():
        raise ApiError(422, f"'{key}' must be a non-empty string")
    if len(t) > max_len:
        raise ApiError(422, f"'{key}' exceeds {max_len} characters")
    return t


def health(_=None):
    e = get_engine()
    return {"status": "ok", "data_source": e.data_source, "embedder": e.embedder.label, "n_faculty": len(e.ids)}


def find_related_researchers(body):
    e = get_engine()
    fid = body.get("faculty_id")
    if not isinstance(fid, str):
        raise ApiError(422, "'faculty_id' (string) is required")
    method = body.get("method", "hybrid")
    if method not in METHODS:
        raise ApiError(422, f"'method' must be one of {list(METHODS)}")
    k = _int(body.get("k"), "k", 1, 20, 3)
    try:
        return e.find_related(fid, k=k, method=method)
    except KeyError:
        raise ApiError(404, f"Unknown faculty_id '{fid}'")


def search_researchers(body):
    e = get_engine()
    method = body.get("method", "hybrid")
    if method not in ("tfidf", "embedding", "hybrid"):
        raise ApiError(422, "'method' must be one of ['tfidf','embedding','hybrid']")
    return e.search(_text(body), k=_int(body.get("k"), "k", 1, 20, 5), method=method)


def extract_keywords(body):
    e = get_engine()
    return {"input_id": body.get("input_id"), "result": {"keywords": e.extract_keywords(_text(body), _int(body.get("n"), "n", 1, 30, 8))}}


def extract_entities(body):
    return {"input_id": body.get("input_id"), "result": get_engine().extract_entities(_text(body))}


def list_faculty(_=None):
    return {"faculty": get_engine().list_faculty()}


def topics(_=None):
    return get_engine().topic_summary()


def knowledge_graph(_=None):
    return get_engine().kg_json()


ROUTES = {
    ("GET", "/health"): health, ("GET", "/faculty"): list_faculty, ("GET", "/topics"): topics,
    ("GET", "/knowledge-graph"): knowledge_graph,
    ("POST", "/find-related-researchers"): find_related_researchers, ("POST", "/search-researchers"): search_researchers,
    ("POST", "/extract-keywords"): extract_keywords, ("POST", "/extract-entities"): extract_entities,
}
