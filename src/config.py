"""Central configuration: paths and defaults."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
DOCS_DIR = os.path.join(ROOT, "docs")

REAL_DATA = os.path.join(DATA_DIR, "faculty_research.csv")          # real, collected data (preferred)
SAMPLE_DATA = os.path.join(DATA_DIR, "sample_faculty_research.csv")  # SYNTHETIC development data
SAMPLE_PAIRS = os.path.join(DATA_DIR, "sample_relevance_pairs.csv")
REAL_PAIRS = os.path.join(DATA_DIR, "relevance_pairs.csv")

SENTENCE_MODEL = os.environ.get("RDCE_SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RANDOM_STATE = 42
DEFAULT_ALPHA = 0.5   # hybrid weight on semantic similarity (tuned on dev split in run_eval.py)


def resolve_data_path():
    """Real data wins; otherwise fall back to the synthetic sample (and say so)."""
    env = os.environ.get("RDCE_DATA")
    if env:
        return env, "CUSTOM"
    if os.path.exists(REAL_DATA):
        return REAL_DATA, "REAL"
    return SAMPLE_DATA, "SYNTHETIC"
