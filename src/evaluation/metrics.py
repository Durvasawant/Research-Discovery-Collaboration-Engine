"""Ranking metrics for researcher retrieval. `rel` is a graded relevance vector (0/1/2) aligned with the ranked list."""
import numpy as np


def precision_at_k(hits, k):
    return float(np.sum(hits[:k]) / k)


def recall_at_k(hits, n_rel, k):
    return float(np.sum(hits[:k]) / n_rel) if n_rel else 0.0


def reciprocal_rank(hits):
    return float(1 / (np.argmax(hits) + 1)) if np.any(hits) else 0.0


def average_precision(hits):
    if not np.any(hits):
        return 0.0
    cum = np.cumsum(hits)
    return float(np.sum((cum / (np.arange(len(hits)) + 1)) * hits) / np.sum(hits))


def ndcg_at_k(rel_ranked, ideal_rel, k):
    gain = lambda r: (2.0 ** np.asarray(r[:k], dtype=float) - 1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gain(rel_ranked) * disc[: len(rel_ranked[:k])]))
    ideal = np.sort(ideal_rel)[::-1]
    idcg = float(np.sum(gain(ideal) * disc[: len(ideal[:k])]))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_metrics(S, REL, queries, ks=(1, 3, 5), threshold=1):
    """Returns dict metric -> np.array over queries that have at least one relevant researcher."""
    out, kept = {}, []
    for i in queries:
        mask = np.arange(len(S)) != i
        order = np.argsort(-S[i])
        order = order[order != i]
        rel_ranked = REL[i][order]
        hits = (rel_ranked >= threshold).astype(int)
        n_rel = int(((REL[i] >= threshold) & mask).sum())
        if n_rel == 0:
            continue
        kept.append(i)
        for k in ks:
            out.setdefault(f"P@{k}", []).append(precision_at_k(hits, k))
            out.setdefault(f"R@{k}", []).append(recall_at_k(hits, n_rel, k))
            out.setdefault(f"nDCG@{k}", []).append(ndcg_at_k(rel_ranked, REL[i][mask], k))
        out.setdefault("MRR", []).append(reciprocal_rank(hits))
        out.setdefault("MAP", []).append(average_precision(hits))
    return {m: np.array(v) for m, v in out.items()}, kept


def bootstrap_ci(values, n_boot=1000, seed=42, level=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def summarise(per_q, with_ci=True):
    row = {}
    for m, v in per_q.items():
        row[m] = round(float(v.mean()), 3)
        if with_ci and m in ("P@3", "MRR", "nDCG@3"):
            lo, hi = bootstrap_ci(v)
            row[m + "_CI95"] = f"[{lo:.2f}, {hi:.2f}]"
    return row
