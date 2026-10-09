"""NER + information extraction.

1. Research-lexicon matcher (METHOD / TASK / DOMAIN) -- general NER models do not know 'U-Net' or 'entity linking'.
2. Optional spaCy general NER (ORG, GPE, ...) when en_core_web_sm is installed.
3. Rule-based triple extraction: (paper, USES, method) and (paper, ADDRESSES, task) used to build the knowledge graph.
"""
import re
from src.preprocessing.text import HAS_SPACY, _NLP

LEXICON = {
 "METHOD": ["transformer", "bert", "cnn", "3d cnn", "lstm", "arima", "gan", "diffusion", "autoencoder", "u-net", "lora",
            "gradient boosting", "logistic regression", "matrix factorisation", "shap", "contrastive learning",
            "self-supervised learning", "federated learning", "adapter", "chain-of-thought", "dense retrieval",
            "large language model", "llm", "rag", "retrieval-augmented generation", "graph embedding", "sentence embedding",
            "tf-idf", "sparql", "owl", "rdf", "smart contract", "tree ensemble", "prompting", "fine-tuning"],
 "TASK": ["named entity recognition", "ner", "relation extraction", "entity linking", "information extraction",
          "sentiment analysis", "text classification", "summarisation", "question answering", "link prediction",
          "semantic search", "object detection", "segmentation", "object tracking", "action recognition",
          "image captioning", "intrusion detection", "anomaly detection", "time series forecasting", "community detection",
          "recommender systems", "dropout prediction", "disease prediction", "semantic textual similarity", "reasoning",
          "hallucination", "malware", "phishing", "prompt injection"],
 "DOMAIN": ["knowledge graph", "ontology", "healthcare", "clinical", "biomedical", "education", "student", "campus",
            "iot", "edge computing", "wireless sensor", "cybersecurity", "finance", "agriculture", "video", "mri", "x-ray",
            "generative ai", "multilingual", "blockchain", "privacy"],
}
_PATTERNS = {
    cat: re.compile(r"(?<![\w-])(" + "|".join(re.escape(x) for x in sorted(words, key=len, reverse=True)) + r")(?:s|es)?(?![\w-])", re.I)
    for cat, words in LEXICON.items()
}


def extract_entities(text: str) -> dict:
    out = {cat.lower() + "s": sorted({m.lower() for m in pat.findall(text)}) for cat, pat in _PATTERNS.items()}
    out["general_entities"] = sorted({(e.text, e.label_) for e in _NLP(text).ents}) if HAS_SPACY else []
    out["general_entities"] = [list(x) for x in out["general_entities"]]
    return out


def extract_triples(paper_id: str, text: str):
    """Rule-based information extraction -> list of (subject, predicate, object)."""
    ent = extract_entities(text)
    return ([(paper_id, "USES", m) for m in ent["methods"]] +
            [(paper_id, "ADDRESSES", t) for t in ent["tasks"]] +
            [(paper_id, "IN_DOMAIN", d) for d in ent["domains"]])
