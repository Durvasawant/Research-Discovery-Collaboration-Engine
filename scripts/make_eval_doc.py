"""Regenerate docs/evaluation.md from results/ so the document can never disagree with the numbers.
    python -m src.evaluation.run_eval && python scripts/make_eval_doc.py"""
import json, os, pandas as pd
R = os.path.join(os.path.dirname(__file__), "..", "results"); D = os.path.join(os.path.dirname(__file__), "..", "docs")
s = json.load(open(f"{R}/evaluation_summary.json"))
def md(df): return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()
try: import tabulate  # noqa
except ImportError:
    def md(df):
        cols = [df.index.name or ""] + list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for i, r in df.iterrows(): out.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
        return "\n".join(out)
t1 = pd.read_csv(f"{R}/results_test_thr1.csv", index_col=0); t2 = pd.read_csv(f"{R}/results_test_thr2.csv", index_col=0)
cols = ["P@1", "P@3", "R@3", "P@5", "nDCG@3", "nDCG@3_CI95", "MRR", "MAP"]
rob = pd.read_csv(f"{R}/robustness_profile_modes.csv").set_index(["profile_text", "method"])
err = pd.read_csv(f"{R}/error_analysis.csv")
synthetic = "SYNTHETIC" in s["banner"]; lsa = "LSA" in s["banner"]
lines = [f"# Evaluation Report (auto-generated)\n", f"`{s['banner']}`\n"]
if synthetic: lines.append("> **WARNING - SYNTHETIC DATA.** Numbers validate the pipeline only. Labels were derived from area tags written together with the data. Do **not** report as final results.\n")
if lsa: lines.append("> **WARNING - no transformer installed.** Rows named `embedding` use the LSA fallback (TF-IDF + SVD), not sentence-transformers. Install `sentence-transformers` and re-run to obtain the real Approach-B numbers.\n")
lines += ["## Protocol", "- Queries = researchers; split 50/50 into **dev/test** (stratified by department, seed 42). The only tuned hyper-parameter (hybrid alpha) is chosen on dev by nDCG@3; all tables below are **test** queries.",
          "- Relevance is graded 0/1/2. *Loose* = label >= 1 counts as relevant; *strict* = label 2 only. Metrics: P@K, R@K, nDCG@K (gain 2^rel-1), MRR, MAP; 95% bootstrap CI (1000 resamples over queries).",
          f"- Alpha tuning on dev (nDCG@3): `{s['alpha_tuning_dev']}` -> alpha = **{s['best_alpha']}**.", f"- Dataset: `{s['dataset']}`\n",
          "## Baseline vs improved (test) - loose relevance", md(t1[cols]), "", "## Strict relevance (label 2 only)", md(t2[cols]), "",
          "![comparison](../results/comparison.png)\n", "## Robustness: how much text is needed? (test, nDCG@3 / P@3 / MRR)", md(rob), "",
          "## Keywords and topics", f"- Keyword hit-rate against author-supplied keywords (top-8): `{s['keyword_hit_rate']}` (circular for TF-IDF, which sees the keywords as text - treat as a sanity check only).",
          f"- NMF topics: k = {s['topics']['n_topics']} chosen by NPMI coherence = {s['topics']['coherence_npmi']} (k-selection: `{s['topics']['k_selection']}`).\n",
          f"## Error analysis (top-3 hybrid false positives; {s['n_false_positives_top3']} in total)", md(err[["query", "wrong_match", "hybrid_score", "tfidf_rank", "embedding_rank", "shared_terms", "likely_cause"]]), "",
          "## How to read these results", "- Overlapping bootstrap CIs mean differences between methods are **not statistically established** with this many queries; say so in the report rather than declaring a winner.",
          "- Typical failure: researchers who append generic ML vocabulary (deep learning, embeddings, benchmark) get matched to unrelated areas by *every* method - add domain-specific stop-lists or weight curated keywords.",
          "- Paraphrase-heavy profiles are where transformer embeddings are expected to beat lexical methods; verify on real data."]
open(f"{D}/evaluation.md", "w", encoding="utf-8").write("\n".join(lines)); print("wrote docs/evaluation.md")
