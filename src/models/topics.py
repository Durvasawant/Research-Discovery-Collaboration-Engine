"""Topic modelling with NMF over paper-level TF-IDF, with coherence-based selection of the number of topics."""
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import RANDOM_STATE
from src.preprocessing.text import preprocess


def npmi_coherence(topic_words, docs_tokens, eps=1e-12):
    """Mean NPMI over the top-word pairs of each topic, estimated from document co-occurrence."""
    sets = [set(d) for d in docs_tokens]
    N = len(sets)
    def p(*ws):
        return sum(all(w in s for w in ws) for s in sets) / N
    scores = []
    for words in topic_words:
        pair = []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                pa, pb, pab = p(words[a]), p(words[b]), p(words[a], words[b])
                if pab == 0:
                    pair.append(-1.0)
                else:
                    pair.append(np.log(pab / (pa * pb + eps) + eps) / (-np.log(pab + eps)))
        scores.append(np.mean(pair) if pair else 0)
    return float(np.mean(scores))


class TopicModel:
    def __init__(self, k=None, k_range=range(6, 13), top_words=8):
        self.k, self.k_range, self.top_words = k, k_range, top_words
        self.selection = {}

    def fit(self, paper_texts):
        clean = [preprocess(t) for t in paper_texts]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
        X = self.vec.fit_transform(clean)
        self.terms = np.array(self.vec.get_feature_names_out())
        docs_tokens = [c.split() for c in clean]
        def fit_k(k):
            m = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=600)
            W = m.fit_transform(X)
            words = [list(self.terms[c.argsort()[::-1][: self.top_words]]) for c in m.components_]
            return m, W, words
        if self.k is None:
            best = None
            for k in self.k_range:
                if k >= X.shape[0]:
                    continue
                m, W, words = fit_k(k)
                # coherence on unigram words (bigrams are not in docs_tokens)
                uni = [[w for w in ws if " " not in w][:6] for ws in words]
                coh = npmi_coherence(uni, docs_tokens)
                self.selection[k] = round(coh, 4)
                if best is None or coh > best[0]:
                    best = (coh, k)
            self.k = best[1]
        self.model, self.W_papers, self.topic_words = fit_k(self.k)
        self.coherence = self.selection.get(self.k)
        return self

    def transform(self, texts):
        W = self.model.transform(self.vec.transform([preprocess(t) for t in texts]))
        return W / (W.sum(1, keepdims=True) + 1e-12)

    def label(self, t, n=3):
        return " / ".join(self.topic_words[t][:n])
