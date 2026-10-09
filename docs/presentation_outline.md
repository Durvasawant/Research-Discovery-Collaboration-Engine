# 10-slide outline (10 minutes)
1. **Title** - Research Discovery & Collaboration Engine, team, AI University context.
2. **Problem** - faculty with related work don't know each other; manual search over profiles is slow.
3. **Use case / users** - faculty, research scholars, departments, research office. Show the required output (related researchers + common topics).
4. **Data** - source, #faculty/#papers, labels (2 annotators, kappa), split, privacy. *(Real numbers!)*
5. **NLP approach** - baseline TF-IDF/BM25 -> sentence-transformer (mean of paper vectors) -> hybrid; keywords, NMF topics, NER lexicon, KG.
6. **Architecture** - `docs/architecture.png`.
7. **Results** - baseline vs improved table + CI, robustness chart (`results/comparison.png`). State honestly if CIs overlap.
8. **Demo** - Streamlit: pick faculty -> related researchers, shared topics, KG; then **one failure** (trap faculty matched by generic ML vocabulary; `results/error_analysis.csv`).
9. **Reusable asset** - `curl` call, `demo/client_example.py`, integration with Group 8 / Group 4.
10. **Limitations & future work** - bias toward prolific authors, small labelled set, lexicon NER, SPECTER2/SciBERT, SciSpaCy, auth + rate limiting, scheduled data refresh.
