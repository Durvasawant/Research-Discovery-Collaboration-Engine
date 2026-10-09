"""Synthetic-data diagnostic: accuracy by faculty type (plain / boilerplate-trap / paraphrased). Meaningful ONLY for the shipped
synthetic sample, where the type is encoded in the faculty index (see generate_sample_data.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import SAMPLE_DATA, SAMPLE_PAIRS
from src.preprocessing.data import load_dataframe
from src.models.engine import ResearchDiscoveryEngine
from src.models.embedder import Embedder
from src.evaluation.ground_truth import from_pairs
from src.evaluation.metrics import per_query_metrics
e = ResearchDiscoveryEngine(load_dataframe(SAMPLE_DATA), embedder=Embedder("auto"), alpha=0.8)
R = from_pairs(e.ids, SAMPLE_PAIRS)
g = {"plain": [i for i, f in enumerate(e.ids) if int(f[1:]) % 4 in (1, 2)], "trap": [i for i, f in enumerate(e.ids) if int(f[1:]) % 4 == 3],
     "paraphrase": [i for i, f in enumerate(e.ids) if int(f[1:]) % 4 == 0]}
print("embedder:", e.embedder.label)
for name, q in g.items():
    print(f"{name:11s} n={len(q):2d}", {m: (round(float(per_query_metrics(e.matrix(m), R, q, threshold=2)[0]['P@3'].mean()), 2),
          round(float(per_query_metrics(e.matrix(m), R, q)[0]['nDCG@3'].mean()), 2)) for m in ["tfidf", "bm25", "embedding", "hybrid"]}, "(strict P@3, nDCG@3)")
