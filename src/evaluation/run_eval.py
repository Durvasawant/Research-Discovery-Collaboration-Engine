"""End-to-end evaluation: baseline vs improved vs hybrid, ablations, robustness, error analysis.

    python -m src.evaluation.run_eval                      # uses real data if present, else synthetic sample
    python -m src.evaluation.run_eval --data data/faculty_research.csv --pairs data/relevance_pairs.csv

Protocol: researchers (queries) are split 50/50 into dev/test, stratified by area when available. The only tuned
hyper-parameter (hybrid alpha) is chosen on DEV; all headline numbers are on TEST.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, RANDOM_STATE, resolve_data_path, SAMPLE_PAIRS, REAL_PAIRS
from src.preprocessing.data import load_dataframe, dataset_stats
from src.models.engine import ResearchDiscoveryEngine
from src.models.embedder import Embedder
from src.models.keywords import tfidf_keywords, embedding_keywords, curated_keywords
from src.evaluation.ground_truth import build_relevance
from src.evaluation.metrics import per_query_metrics, summarise

ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def split_queries(engine, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(engine.researchers):
        groups.setdefault(r.department, []).append(i)      # stratify by department (always available)
    dev, test = [], []
    for g in groups.values():
        g = list(rng.permutation(g)); h = len(g) // 2
        dev += g[:h]; test += g[h:]
    return sorted(dev), sorted(test)


def evaluate_engine(engine, REL, queries, threshold=1, alpha=None):
    rows = {}
    for name, S in {"tfidf (baseline)": engine.matrix("tfidf"), "bm25": engine.matrix("bm25"),
                    "embedding (concat profile)": engine.matrix("embedding_concat"),
                    "embedding (mean of docs)": engine.matrix("embedding"),
                    f"hybrid (alpha={engine.alpha if alpha is None else alpha})": engine.matrix("hybrid", alpha)}.items():
        pq, kept = per_query_metrics(S, REL, queries, threshold=threshold)
        rows[name] = {**summarise(pq), "n_queries": len(kept)}
    return pd.DataFrame(rows).T


def error_analysis(engine, REL, S_name="hybrid", k=3, limit=12):
    S, St, Se = engine.matrix(S_name), engine.matrix("tfidf"), engine.matrix("embedding")
    rank_of = lambda M, i, j: int((np.argsort(-M[i]).tolist().index(j)))   # 0-based (diagonal is 0 so it sits last)
    errs = []
    for i in range(len(engine.ids)):
        order = [j for j in np.argsort(-S[i]) if j != i][:k]
        for pos, j in enumerate(order):
            if REL[i, j] == 0:
                rt, re_ = rank_of(St, i, j), rank_of(Se, i, j)
                cause = ("lexical overlap: shared surface vocabulary, different research area" if rt < k and re_ >= k else
                         "semantic over-generalisation: embedding places different areas close" if re_ < k and rt >= k else
                         "both lexical and semantic signals mislead (shared generic ML vocabulary / bridge topics)")
                errs.append({"query": engine.ids[i], "wrong_match": engine.ids[j], "rank": pos + 1,
                             "hybrid_score": round(float(S[i, j]), 3), "tfidf_rank": rt + 1, "embedding_rank": re_ + 1,
                             "shared_terms": ", ".join(engine._shared_terms(i, j, 5)),
                             "query_interests": engine.researchers[i].interests,
                             "match_interests": engine.researchers[j].interests, "likely_cause": cause})
    if not errs:
        return pd.DataFrame(columns=["query", "wrong_match", "hybrid_score", "shared_terms", "likely_cause"]), 0
    return pd.DataFrame(errs).sort_values("hybrid_score", ascending=False).head(limit).reset_index(drop=True), len(errs)


def keyword_hit_rate(engine, n=8):
    """Share of the extracted top-n keywords that overlap (substring either way) an author-supplied keyword."""
    res = {"tfidf": [], "embedding (MMR)": []}
    for i, r in enumerate(engine.researchers):
        cur = curated_keywords(r)
        if not cur: continue
        hit = lambda ks: np.mean([any(c in k or k in c for c in cur) for k in ks]) if ks else 0
        res["tfidf"].append(hit(tfidf_keywords(engine.vec, engine.X, i, n)))
        res["embedding (MMR)"].append(hit(engine.keywords(r.faculty_id, n)))
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--pairs"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default=RESULTS_DIR)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path, label = (a.data, "CUSTOM") if a.data else resolve_data_path()
    pairs = a.pairs or (REAL_PAIRS if os.path.exists(REAL_PAIRS) else (SAMPLE_PAIRS if label == "SYNTHETIC" else None))
    df = load_dataframe(path)
    emb = Embedder(a.backend)
    eng = ResearchDiscoveryEngine(df, embedder=emb, data_source=label)
    REL, rel_src = build_relevance(eng.researchers, eng.ids, pairs)
    dev, test = split_queries(eng)
    banner = (f"DATA={label} | EMBEDDER={emb.label} | relevance={rel_src} | queries dev/test={len(dev)}/{len(test)}")
    print(banner)
    if label == "SYNTHETIC": print("!! SYNTHETIC DATA: numbers below are for pipeline validation only, NOT final results.")
    if emb.backend != "sentence-transformer": print("!! No transformer available: 'embedding' rows are LSA, not transformer embeddings.")

    # tune alpha on DEV only
    tune = {al: summarise(per_query_metrics(eng.matrix("hybrid", al), REL, dev)[0], False)["nDCG@3"] for al in ALPHAS}
    best_alpha = max(tune, key=tune.get)
    eng.alpha = best_alpha
    print("alpha tuning on DEV (nDCG@3):", tune, "-> alpha =", best_alpha)

    out = {"banner": banner, "dataset": dataset_stats(df), "alpha_tuning_dev": tune, "best_alpha": best_alpha}
    for thr, nm in [(1, "relevant>=1 (related or partially related)"), (2, "relevant==2 (strict)")]:
        t = evaluate_engine(eng, REL, test, thr, best_alpha)
        t.to_csv(f"{a.out}/results_test_thr{thr}.csv"); out[f"test_{nm}"] = t.reset_index().to_dict("records")
        print(f"\nTEST  {nm}\n", t.drop(columns=[c for c in t.columns if c.endswith('CI95')]).to_string())
    allq = evaluate_engine(eng, REL, list(range(len(eng.ids))), 1, best_alpha); allq.to_csv(f"{a.out}/results_allqueries_thr1.csv")

    # robustness: how much text does the system need?
    rob = {}
    for mode in ("full", "titles_keywords", "interests_only"):
        e2 = ResearchDiscoveryEngine(df, embedder=Embedder(a.backend), profile_mode=mode, alpha=best_alpha, data_source=label)
        R2, _ = build_relevance(e2.researchers, e2.ids, pairs)
        d2, t2 = split_queries(e2)
        for name in ("tfidf", "embedding", "hybrid"):
            rob[(mode, name)] = summarise(per_query_metrics(e2.matrix(name), R2, t2)[0], False)
    rob_df = pd.DataFrame(rob).T[["P@3", "R@3", "nDCG@3", "MRR"]]; rob_df.index.names = ["profile_text", "method"]
    rob_df.to_csv(f"{a.out}/robustness_profile_modes.csv"); print("\nROBUSTNESS (TEST)\n", rob_df.to_string())
    out["robustness"] = rob_df.reset_index().to_dict("records")

    kh = keyword_hit_rate(eng); out["keyword_hit_rate"] = kh; print("\nKeyword hit rate vs author keywords:", kh)
    out["topics"] = eng.topic_summary(); print("Topics: k =", eng.topics.k, "NPMI =", eng.topics.coherence)
    errs, n_err = error_analysis(eng, REL)
    errs.to_csv(f"{a.out}/error_analysis.csv", index=False); out["n_false_positives_top3"] = n_err
    print(f"\n{n_err} unrelated researchers appear in a top-3 list; worst:\n", errs[["query", "wrong_match", "hybrid_score", "shared_terms", "likely_cause"]].head(6).to_string())

    # plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t1 = pd.read_csv(f"{a.out}/results_test_thr1.csv", index_col=0)[["P@3", "R@3", "nDCG@3", "MRR"]]
    ax = t1.plot(kind="bar", figsize=(9, 4.5), rot=15); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"Test-set comparison ({label} data, {emb.backend})"); plt.tight_layout(); plt.savefig(f"{a.out}/comparison.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.plot(list(tune), list(tune.values()), "o-"); ax.set_xlabel("alpha (weight on embeddings)")
    ax.set_ylabel("DEV nDCG@3"); ax.grid(alpha=.3); ax.set_title("Hybrid weight tuning (dev)"); plt.tight_layout(); plt.savefig(f"{a.out}/alpha_tuning.png", dpi=150); plt.close()
    json.dump(out, open(f"{a.out}/evaluation_summary.json", "w"), indent=2, default=str)
    # demo outputs + knowledge graph artefacts
    demo = {fid: eng.find_related(fid, 3) for fid in eng.ids}
    json.dump(demo, open(f"{a.out}/sample_outputs.json", "w"), indent=2)
    json.dump(eng.kg_json(), open(f"{a.out}/knowledge_graph.json", "w"), indent=2)
    from src.models.kg import draw_ego
    draw_ego(eng.knowledge_graph(), eng.ids[4], f"{a.out}/knowledge_graph_example.png")
    print("\nSaved results to", a.out)


if __name__ == "__main__":
    main()
