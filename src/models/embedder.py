"""Document embedder with a transparent fallback.

backend="sentence-transformer" -> real transformer embeddings (all-MiniLM-L6-v2 by default; SPECTER2/SciBERT can be
                                  supplied through model_name).
backend="lsa"                  -> TF-IDF + truncated SVD (Latent Semantic Analysis). Used ONLY when the transformer
                                  cannot be loaded. Results from it must be labelled "LSA", never "transformer".
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import SENTENCE_MODEL, RANDOM_STATE
from src.preprocessing.text import preprocess


class Embedder:
    def __init__(self, backend: str = "auto", model_name: str = SENTENCE_MODEL, lsa_components: int = 24):
        self.model_name, self.lsa_components = model_name, lsa_components
        self.backend, self._st = None, None
        if backend in ("auto", "sentence-transformer"):
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
                self.backend = "sentence-transformer"
            except Exception as e:
                if backend == "sentence-transformer":
                    raise RuntimeError(f"Could not load {model_name}: {e}")
        if self.backend is None:
            self.backend = "lsa"
        self._tfidf = self._svd = None

    @property
    def label(self) -> str:
        return f"sentence-transformer:{self.model_name}" if self.backend == "sentence-transformer" else "LSA-fallback"

    def fit(self, corpus):
        """Needed only for LSA (a no-op for the transformer). `corpus` = list of raw texts."""
        if self.backend == "lsa":
            docs = [preprocess(t) for t in corpus]
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
            X = self._tfidf.transform(docs)
            k = max(2, min(self.lsa_components, X.shape[0] - 1, X.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE).fit(X)
        return self

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.backend == "sentence-transformer":
            v = np.asarray(self._st.encode(texts, normalize_embeddings=True, show_progress_bar=False))
            return v
        if self._svd is None:
            raise RuntimeError("LSA embedder must be fit() before encode().")
        v = self._svd.transform(self._tfidf.transform([preprocess(t) for t in texts]))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
