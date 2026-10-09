"""Similarity models. Every function returns an (n x n) matrix with a zeroed diagonal
(a researcher is never 'related' to themself)."""
import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.text import preprocess


def _zero_diag(S):
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, 0.0)
    return S


def tfidf_matrix(clean_docs, **kw):
    """Approach A (baseline): TF-IDF (uni+bi-grams) + cosine. Returns (S, vectorizer, X)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, **kw)
    X = vec.fit_transform(clean_docs)
    return _zero_diag(cosine_similarity(X)), vec, X


def bm25_matrix(clean_docs, k1=1.5, b=0.75):
    """Second lexical baseline. Each profile is used as a query against all profiles; scores are divided by the
    query's self-score (so 1.0 = identical) and symmetrised."""
    toks = [d.split() for d in clean_docs]
    N = len(toks)
    avgdl = np.mean([len(t) for t in toks]) or 1.0
    df = Counter(w for t in toks for w in set(t))
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    tfs = [Counter(t) for t in toks]

    def score(q, j):
        dl, s = len(toks[j]), 0.0
        for w in set(toks[q]):
            f = tfs[j].get(w, 0)
            if f:
                s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        return s

    S = np.zeros((N, N))
    for i in range(N):
        selfs = score(i, i) or 1.0
        for j in range(N):
            S[i, j] = score(i, j) / selfs
    S = (S + S.T) / 2
    return _zero_diag(np.clip(S, 0, 1))


def embedding_matrix(E):
    """Approach B: cosine over (L2-normalised) document embeddings."""
    return _zero_diag(np.clip(cosine_similarity(E), 0, 1))


def mean_pool(embedder, researchers, mode="full"):
    """Researcher vector = mean of the embeddings of each of their documents (interests + each paper).
    Avoids the 256/512-token truncation that hurts concatenated profiles and weights every paper equally."""
    vecs = []
    for r in researchers:
        v = embedder.encode(r.documents(mode)).mean(axis=0)
        vecs.append(v / (np.linalg.norm(v) + 1e-12))
    return np.vstack(vecs)


def hybrid_matrix(S_lex, S_sem, alpha=0.5):
    """alpha * semantic + (1 - alpha) * lexical."""
    return _zero_diag(alpha * S_sem + (1 - alpha) * S_lex)


def rank(S, i, k):
    order = np.argsort(-S[i])
    order = order[order != i]
    return order[:k]
