"""ResearchDiscoveryEngine: the reusable NLP asset. One object wires preprocessing, representation, similarity,
keyword extraction, topic modelling, NER/IE and the knowledge graph behind a small, JSON-friendly API."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_ALPHA, RANDOM_STATE
from src.preprocessing.data import build_researchers, validate, PROFILE_MODES
from src.preprocessing.text import preprocess
from src.models.embedder import Embedder
from src.models import similarity as sim
from src.models.keywords import tfidf_keywords, embedding_keywords, curated_keywords
from src.models.topics import TopicModel
from src.models.entities import extract_entities
from src.models import kg as kgmod

METHODS = ("tfidf", "bm25", "embedding", "hybrid")


class ResearchDiscoveryEngine:
    def __init__(self, df: pd.DataFrame, embedder: Optional[Embedder] = None, alpha: float = DEFAULT_ALPHA,
                 profile_mode: str = "full", n_topics: Optional[int] = None, data_source: str = "UNKNOWN"):
        if profile_mode not in PROFILE_MODES:
            raise ValueError(f"profile_mode must be one of {PROFILE_MODES}")
        self.df = validate(df)
        self.data_source, self.alpha, self.profile_mode = data_source, alpha, profile_mode
        self.researchers = build_researchers(self.df)
        self.ids = [r.faculty_id for r in self.researchers]
        self.index = {f: i for i, f in enumerate(self.ids)}
        self.raw = [r.profile_text(profile_mode) for r in self.researchers]
        self.clean = [preprocess(t) for t in self.raw]

        # --- representations -------------------------------------------------
        self.S_tfidf, self.vec, self.X = sim.tfidf_matrix(self.clean)
        self.S_bm25 = sim.bm25_matrix(self.clean)
        self.embedder = embedder or Embedder()
        corpus = [d for r in self.researchers for d in r.documents(profile_mode)] + self.raw
        self.embedder.fit(corpus)
        self.E = sim.mean_pool(self.embedder, self.researchers, profile_mode)       # mean of per-document embeddings
        self.E_concat = self.embedder.encode(self.raw)                              # concatenated-profile variant (ablation)
        self.S_emb = sim.embedding_matrix(self.E)
        self.S_emb_concat = sim.embedding_matrix(self.E_concat)

        # --- topics (paper-level NMF) ---------------------------------------
        self._paper_ids, paper_texts = [], []
        pi = 0
        for r in self.researchers:
            for p in r.papers:
                self._paper_ids.append(f"{r.faculty_id}-P{pi}"); paper_texts.append(p.text); pi += 1
        self.topics = TopicModel(k=n_topics).fit(paper_texts)
        Wp = self.topics.W_papers / (self.topics.W_papers.sum(1, keepdims=True) + 1e-12)
        self.paper_topic = dict(zip(self._paper_ids, Wp))
        Wf = []
        for r in self.researchers:
            rows = [self.paper_topic[pid] for pid in self._paper_ids if pid.startswith(r.faculty_id + "-")]
            w = np.mean(rows, axis=0) if rows else np.zeros(self.topics.k)
            Wf.append(w / (w.sum() + 1e-12))
        self.W = np.vstack(Wf)
        self._graph = None

    # ----------------------------------------------------------------- matrices
    def matrix(self, method: str, alpha: Optional[float] = None) -> np.ndarray:
        if method == "tfidf": return self.S_tfidf
        if method == "bm25": return self.S_bm25
        if method == "embedding": return self.S_emb
        if method == "embedding_concat": return self.S_emb_concat
        if method == "hybrid": return sim.hybrid_matrix(self.S_tfidf, self.S_emb, self.alpha if alpha is None else alpha)
        raise ValueError(f"method must be one of {METHODS}")

    # ----------------------------------------------------------------- explanations
    def _shared_curated(self, i, j, top=6):
        a, b = curated_keywords(self.researchers[i]), curated_keywords(self.researchers[j])
        sa = set(a); return [k for k in b if k in sa][:top]

    def _shared_topics(self, i, j, top=3, min_strength=0.15):
        ov = np.minimum(self.W[i], self.W[j])
        return [{"topic_id": int(t), "label": self.topics.label(t), "top_terms": self.topics.topic_words[t][:4],
                 "strength": round(float(ov[t]), 3)} for t in ov.argsort()[::-1][:top] if ov[t] >= min_strength]

    def _shared_terms(self, i, j, top=6):
        terms = np.array(self.vec.get_feature_names_out())
        prod = self.X[i].multiply(self.X[j]).toarray().ravel()
        return [str(t) for t in terms[prod.argsort()[::-1][:top]] if prod[terms.tolist().index(t)] > 0]

    # ----------------------------------------------------------------- public API
    def find_related(self, faculty_id: str, k: int = 3, method: str = "hybrid") -> Dict:
        """Related researchers + common topics for one faculty member (the handout's required output)."""
        if faculty_id not in self.index:
            raise KeyError(f"Unknown faculty_id '{faculty_id}'")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        i, S = self.index[faculty_id], self.matrix(method)
        related, common = [], {}
        for j in sim.rank(S, i, k):
            sk = self._shared_curated(i, j)
            st = self._shared_topics(i, j)
            for x in sk: common[x] = common.get(x, 0) + 1
            related.append({"faculty_id": self.ids[j], "faculty_name": self.researchers[j].name,
                            "department": self.researchers[j].department, "similarity": round(float(S[i, j]), 3),
                            "shared_keywords": sk, "shared_terms": self._shared_terms(i, j), "shared_topics": st})
        r = self.researchers[i]
        return {"input_id": faculty_id, "method": method, "embedder": self.embedder.label, "data_source": self.data_source,
                "result": {"researcher": r.name, "department": r.department,
                           "top_keywords": self.keywords(faculty_id, 6),
                           "entities": extract_entities(self.raw[i]),
                           "related_researchers": related,
                           "common_topics": [k_ for k_, _ in sorted(common.items(), key=lambda x: -x[1])][:8]},
                "disclaimer": "Suggestions for discovery only; not an assessment of research quality or a collaboration decision."}

    def search(self, text: str, k: int = 5, method: str = "hybrid") -> Dict:
        """Ad-hoc query (e.g. a student describing their interests) -> best matching faculty."""
        q_clean = preprocess(text)
        s_lex = (self.vec.transform([q_clean]) @ self.X.T).toarray().ravel()
        qv = self.embedder.encode([text])[0]
        s_sem = np.clip(self.E @ qv, 0, 1)
        score = {"tfidf": s_lex, "embedding": s_sem, "hybrid": self.alpha * s_sem + (1 - self.alpha) * s_lex}.get(method)
        if score is None: raise ValueError("method must be tfidf, embedding or hybrid")
        top = np.argsort(-score)[:k]
        return {"query": text, "method": method, "embedder": self.embedder.label,
                "results": [{"faculty_id": self.ids[j], "faculty_name": self.researchers[j].name,
                             "department": self.researchers[j].department, "score": round(float(score[j]), 3),
                             "top_keywords": self.keywords(self.ids[j], 5)} for j in top]}

    def keywords(self, faculty_id: str, n: int = 8, method: str = "embedding") -> List[str]:
        i = self.index[faculty_id]
        if method == "tfidf":
            return tfidf_keywords(self.vec, self.X, i, n)
        cands_text = " . ".join(self.researchers[i].documents(self.profile_mode))
        return embedding_keywords(self.embedder, cands_text, doc_vec=self.E[i], top_n=n)

    def extract_keywords(self, text: str, n: int = 8) -> List[str]:
        return embedding_keywords(self.embedder, text, top_n=n)

    def extract_entities(self, text: str) -> Dict:
        return extract_entities(text)

    def topic_summary(self) -> Dict:
        return {"n_topics": int(self.topics.k), "coherence_npmi": self.topics.coherence,
                "k_selection": self.topics.selection,
                "topics": [{"topic_id": t, "label": self.topics.label(t), "top_terms": self.topics.topic_words[t]}
                           for t in range(self.topics.k)]}

    def list_faculty(self) -> List[Dict]:
        return [{"faculty_id": r.faculty_id, "faculty_name": r.name, "department": r.department,
                 "n_papers": len(r.papers)} for r in self.researchers]

    def knowledge_graph(self):
        if self._graph is None:
            self._graph = kgmod.build_graph(self.researchers, self.topics, self.paper_topic, self.W, self.matrix("hybrid"))
        return self._graph

    def kg_json(self) -> Dict:
        return kgmod.to_json(self.knowledge_graph())

    def kg_facts(self, faculty_id: str) -> List:
        return kgmod.ego_facts(self.knowledge_graph(), faculty_id)


def load_engine(path: Optional[str] = None, **kw) -> ResearchDiscoveryEngine:
    """Convenience loader: real data if present, else the synthetic sample (flagged in every response)."""
    from src.config import resolve_data_path
    from src.preprocessing.data import load_dataframe
    src, label = (path, "CUSTOM") if path else resolve_data_path()
    return ResearchDiscoveryEngine(load_dataframe(src), data_source=label, **kw)
