# Responsible AI Statement

**Purpose.** The engine *suggests* researchers with related interests. It never ranks researchers' quality, never recommends hiring, funding or promotion, and a human always decides whether to collaborate. Every API response carries a disclaimer.

| Area | Risk | Mitigation in this project | Residual risk |
|---|---|---|---|
| **Privacy** | profiles name real faculty; scraping personal data | only public bibliographic metadata; no student data; IDs instead of names in shared artefacts; faculty opt-out/consent before deployment | faculty may not expect indexing; needs institutional policy |
| **Bias** | prolific authors and English-language, well-indexed fields get richer profiles and surface more often; new/interdisciplinary or non-English researchers are under-represented; embedding models encode corpus bias | profile = mean of per-paper vectors (a prolific author is not rewarded by length); curated keywords up-weighted; limitation reported; recommended audit: compare retrieval rates by department/seniority | not measured on real data yet |
| **Hallucination** | none of the components generates text; all outputs are retrieved or extracted from source documents | extractive keywords/NER/triples; explanations are shared terms taken verbatim from the profiles | lexicon NER can mis-tag |
| **Explainability** | black-box similarity scores | each match lists shared keywords, shared terms and shared topics | embedding similarity itself is not decomposable |
| **Security** | malicious text in abstracts (prompt injection if an LLM is later added); API abuse | no LLM in the loop; input length limits and validation; no code execution; (if an LLM summariser is added, screen inputs with Group 12's module) | no authentication/rate limiting in the prototype |
| **Human oversight** | over-trust in a score | scores are relative; disclaimer; users see evidence | users may still over-trust |
| **Limitations** | see `docs/evaluation.md` | | |
