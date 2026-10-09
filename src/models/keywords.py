"""Keyword extraction: (1) TF-IDF, (2) embedding re-ranking (KeyBERT-style)."""
import re
import numpy as np


def tfidf_keywords(vec, X, idx, top_n=10, min_len=3):
    terms = np.array(vec.get_feature_names_out())
    row = X[idx].toarray().ravel()
    order = [j for j in row.argsort()[::-1] if row[j] > 0 and len(terms[j]) >= min_len]
    return [str(terms[j]) for j in order[:top_n]]


def ngram_candidates(text, n_max=3):
    from src.preprocessing.text import STOP
    cands = set()
    # n-grams are generated inside punctuation-delimited segments only, so they never straddle sentence/keyword borders
    for seg in re.split(r"[.;,:()\n]+", str(text).lower()):
        words = re.findall(r"[a-z0-9][a-z0-9\-]*", seg)
        for n in range(1, n_max + 1):
            for i in range(len(words) - n + 1):
                g = words[i:i + n]
                if g[0] in STOP or g[-1] in STOP or any(len(w) < 2 for w in g):
                    continue
                cands.add(" ".join(g))
    return sorted(cands)


def embedding_keywords(embedder, text, doc_vec=None, top_n=8, diversify=True):
    """Rank candidate n-grams by cosine similarity with the document embedding; Maximal Marginal Relevance (MMR)
    removes near-duplicates ('knowledge graph' vs 'knowledge graphs')."""
    cands = ngram_candidates(text)
    if not cands:
        return []
    ce = embedder.encode(cands)
    dv = embedder.encode([text])[0] if doc_vec is None else doc_vec
    sims = ce @ dv
    if not diversify:
        return [cands[i] for i in np.argsort(-sims)[:top_n]]
    chosen, remaining = [], list(range(len(cands)))
    while remaining and len(chosen) < top_n:
        if not chosen:
            best = max(remaining, key=lambda i: sims[i])
        else:
            best = max(remaining, key=lambda i: 0.7 * sims[i] - 0.3 * max(ce[i] @ ce[j] for j in chosen))
        chosen.append(best); remaining.remove(best)
    return [cands[i] for i in chosen]


def curated_keywords(researcher):
    """Author-supplied keywords (normalised) -- the most precise topical signal when available."""
    seen, out = set(), []
    for p in researcher.papers:
        for k in p.keywords:
            kk = k.lower().strip()
            if kk not in seen:
                seen.add(kk); out.append(kk)
    return out
