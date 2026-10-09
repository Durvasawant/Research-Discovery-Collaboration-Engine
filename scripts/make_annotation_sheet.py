"""Create the pair-annotation sheet for human labelling (POOLED evaluation).

Annotating all n(n-1)/2 pairs is infeasible for 40+ faculty, so we pool: the union of every method's top-N candidates
(+ random negatives to estimate recall) is annotated. This avoids favouring the method used to build the benchmark.

    python scripts/make_annotation_sheet.py --top 8 --random 2          -> data/annotation_sheet.csv
Each annotator fills `label_annotator1` / `label_annotator2` independently with 0/1/2 (see docs/annotation_guidelines.md),
then:  python scripts/make_annotation_sheet.py --merge data/annotation_sheet.csv   -> data/relevance_pairs.csv + kappa
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import resolve_data_path, DATA_DIR
from src.preprocessing.data import load_dataframe
from src.models.engine import ResearchDiscoveryEngine
from src.models.embedder import Embedder
from src.evaluation.ground_truth import cohens_kappa


def build(top, n_random, seed=42):
    path, label = resolve_data_path()
    eng = ResearchDiscoveryEngine(load_dataframe(path), embedder=Embedder("auto"), data_source=label)
    rng, pairs = np.random.default_rng(seed), set()
    for m in ("tfidf", "bm25", "embedding", "hybrid"):
        S = eng.matrix(m)
        for i in range(len(eng.ids)):
            for j in np.argsort(-S[i])[: top + 1]:
                if j != i: pairs.add(tuple(sorted((i, int(j)))))
    for i in range(len(eng.ids)):
        for j in rng.choice([x for x in range(len(eng.ids)) if x != i], n_random, replace=False): pairs.add(tuple(sorted((i, int(j)))))
    rows = [{"faculty_id_a": eng.ids[i], "faculty_id_b": eng.ids[j], "interests_a": eng.researchers[i].interests,
             "interests_b": eng.researchers[j].interests, "label_annotator1": "", "label_annotator2": "", "notes": ""} for i, j in sorted(pairs)]
    return pd.DataFrame(rows), len(eng.ids)


def merge(sheet):
    df = pd.read_csv(sheet).dropna(subset=["label_annotator1", "label_annotator2"])
    a, b = df.label_annotator1.astype(int), df.label_annotator2.astype(int)
    print(f"{len(df)} doubly-annotated pairs | raw agreement {np.mean(a == b):.2f} | Cohen's kappa {cohens_kappa(a, b):.2f}")
    print("Disagreements to adjudicate:", int((a != b).sum()))
    out = df[["faculty_id_a", "faculty_id_b"]].copy(); out["label_annotator1"], out["label_annotator2"] = a, b
    out["label"] = np.where(a == b, a, np.ceil((a + b) / 2)).astype(int)   # adjudicate disagreements by hand, then re-run
    out.to_csv(os.path.join(DATA_DIR, "relevance_pairs.csv"), index=False); print("wrote data/relevance_pairs.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=8); ap.add_argument("--random", type=int, default=2); ap.add_argument("--merge")
    a = ap.parse_args()
    if a.merge: merge(a.merge)
    else:
        df, n = build(a.top, a.random); df.to_csv(os.path.join(DATA_DIR, "annotation_sheet.csv"), index=False)
        print(f"{len(df)} pairs to annotate (of {n*(n-1)//2} possible) -> data/annotation_sheet.csv")
