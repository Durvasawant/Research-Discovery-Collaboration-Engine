# API / Module Documentation - Research Discovery & Collaboration Engine

**Reusable-asset test:** *"If another group wanted to use our component tomorrow, what would they need?"*
Python 3.10+, `pip install -r requirements.txt`, then **either** use the Python module **or** start the HTTP service.

## 1. Run the service
```bash
uvicorn src.api.app:app --port 8000            # FastAPI (interactive docs at /docs)
python -m src.api.simple_server --port 8000    # zero-dependency alternative, identical routes
```
Data: `data/faculty_research.csv` if present, else the **synthetic** sample (every response carries `data_source`). Override with env `RDCE_DATA=/path/to.csv`.
Set `RDCE_SENTENCE_MODEL` to swap the embedding model (e.g. `allenai/specter2_base`, `allenai-specter`).

## 2. Python module
```python
from src.models.engine import load_engine
engine = load_engine()                                   # or load_engine("my.csv")
out = engine.find_related("F05", k=3, method="hybrid")   # dict, JSON-serialisable
```

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, data source, embedder in use |
| GET | `/faculty` | list of faculty ids / names / departments |
| POST | `/find-related-researchers` | related researchers + common topics for one faculty member |
| POST | `/search-researchers` | free-text interests -> best matching faculty (students / other groups) |
| POST | `/extract-keywords` | keywords from arbitrary research text |
| POST | `/extract-entities` | methods / tasks / domains (+ general NER if spaCy installed) |
| GET | `/topics` | NMF topics, coherence, k-selection curve |
| GET | `/knowledge-graph` | node-link JSON of the research knowledge graph |

### POST `/find-related-researchers`
| Parameter | Type | Default | Notes |
|---|---|---|---|
| `faculty_id` | string | required | must exist in the dataset (404 otherwise) |
| `k` | int 1-20 | 3 | number of researchers returned |
| `method` | string | `hybrid` | `tfidf` (baseline), `bm25`, `embedding`, `hybrid` |

```bash
curl -X POST localhost:8000/find-related-researchers -H 'Content-Type: application/json' -d '{"faculty_id":"F05","k":3}'
```
Response (abridged):
```json
{
  "input_id": "F05", "method": "hybrid", "embedder": "sentence-transformer:all-MiniLM-L6-v2", "data_source": "REAL",
  "result": {
    "researcher": "Faculty 05", "department": "Computer Science",
    "top_keywords": ["knowledge graph construction", "..."],
    "entities": {"methods": ["graph embedding"], "tasks": ["link prediction"], "domains": ["knowledge graph", "ontology"], "general_entities": []},
    "related_researchers": [
      {"faculty_id": "F06", "faculty_name": "Faculty 06", "department": "Computer Science", "similarity": 0.496,
       "shared_keywords": ["knowledge graph", "ontology", "ner"], "shared_terms": ["knowledge graph", "..."],
       "shared_topics": [{"topic_id": 2, "label": "graph / knowledge / knowledge graph", "top_terms": ["graph","knowledge"], "strength": 0.41}]}
    ],
    "common_topics": ["knowledge graph", "ontology", "ner"]
  },
  "disclaimer": "Suggestions for discovery only; not an assessment of research quality or a collaboration decision."
}
```
**`similarity`** is in [0, 1]; it is a *relative ranking score*, not a probability - do not threshold it across datasets or embedders.

### POST `/search-researchers`
`{"text": "...", "k": 5, "method": "hybrid"}` -> `{"query", "method", "embedder", "results": [{"faculty_id","faculty_name","department","score","top_keywords"}]}`

### POST `/extract-keywords` / `/extract-entities`
`{"text": "...", "input_id": "optional", "n": 8}` -> `{"input_id", "result": {...}}`

## 4. Errors
`422` invalid/missing parameter (message names the field) - `404` unknown `faculty_id` or route - `400` malformed JSON. Body: `{"error": "..."}` (stdlib server) / `{"detail": "..."}` (FastAPI).

## 5. Limits & guarantees
Deterministic for fixed data and model (seeded). Text input <= 20,000 chars. Matrices are computed once at start-up (O(n^2) memory; fine up to a few thousand researchers).
