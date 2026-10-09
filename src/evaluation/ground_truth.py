"""Ground-truth relevance between researchers.

Graded labels: 2 = closely related (same primary area), 1 = partially related, 0 = unrelated.
Sources (in priority order):
  1. a pair-annotation CSV (columns faculty_id_a, faculty_id_b, and either `label` or `label_annotator1`, `label_annotator2`)
  2. a derived rule from the `areas` column of the dataset (ONLY for development/synthetic data)
Unlabelled pairs in a pair-annotation file are treated as 0 -- document this when reporting.
"""
import numpy as np
import pandas as pd


def from_pairs(ids, pairs_csv):
    df = pd.read_csv(pairs_csv)
    ann = [c for c in df.columns if c.startswith("label_annotator")]
    df["_label"] = df[ann].mean(axis=1).round() if ann else df["label"]
    idx = {f: i for i, f in enumerate(ids)}
    R = np.zeros((len(ids), len(ids)), dtype=int)
    for a, b, l in zip(df.faculty_id_a, df.faculty_id_b, df._label):
        if a in idx and b in idx:
            R[idx[a], idx[b]] = R[idx[b], idx[a]] = int(l)
    return R


def from_areas(researchers):
    """Derived labels: 2 if the first-listed... (primary) area sets share >=1 tag AND one is the other's primary; here we
    use: 2 = identical area sets overlap by Jaccard >= 0.5, 1 = any shared tag, 0 = none."""
    n = len(researchers)
    R = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j or researchers[i].areas is None or researchers[j].areas is None:
                continue
            a, b = researchers[i].areas, researchers[j].areas
            inter = len(a & b)
            R[i, j] = 0 if inter == 0 else (2 if inter / len(a | b) >= 0.5 else 1)
    return R


def cohens_kappa(a, b, labels=(0, 1, 2)):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def build_relevance(researchers, ids, pairs_csv=None):
    if pairs_csv:
        return from_pairs(ids, pairs_csv), f"pairs:{pairs_csv}"
    return from_areas(researchers), "derived-from-areas"
