"""Run with:  python -m unittest discover -s tests -v      (pytest also works)"""
import json, os, tempfile, threading, unittest, urllib.request, urllib.error
import numpy as np
import pandas as pd

from src.config import SAMPLE_DATA, SAMPLE_PAIRS
from src.preprocessing.text import preprocess, light_lemma
from src.preprocessing.data import load_dataframe, validate, build_researchers, dataset_stats
from src.models.engine import ResearchDiscoveryEngine
from src.models.embedder import Embedder
from src.models.entities import extract_entities, extract_triples
from src.models import similarity as sim
from src.evaluation import metrics as M
from src.evaluation.ground_truth import from_pairs, cohens_kappa
from src.api import handlers
from src.api.simple_server import make_server

_DF = load_dataframe(SAMPLE_DATA)
ENGINE = ResearchDiscoveryEngine(_DF, embedder=Embedder("lsa"), data_source="TEST")
handlers.set_engine(ENGINE)


class TestPreprocessing(unittest.TestCase):
    def test_stopwords_and_case(self):
        out = preprocess("We propose a Novel approach for Named Entity Recognition")
        self.assertIn("named", out); self.assertNotIn("propose", out); self.assertNotIn("we", out.split())

    def test_lemma_keeps_analysis(self):
        self.assertEqual(light_lemma("analysis"), "analysis"); self.assertEqual(light_lemma("graphs"), "graph")
        self.assertEqual(light_lemma("studies"), "study")

    def test_none_and_empty(self):
        self.assertEqual(preprocess(None), ""); self.assertEqual(preprocess(""), "")

    def test_validate_requires_columns(self):
        with self.assertRaises(ValueError):
            validate(pd.DataFrame({"faculty_id": ["a"]}))

    def test_dedup_and_stats(self):
        df = pd.concat([_DF, _DF.head(3)]); self.assertEqual(len(validate(df)), len(_DF))
        self.assertEqual(dataset_stats(_DF)["faculty"], 32)


class TestSimilarity(unittest.TestCase):
    def test_matrices_shape_diag_range(self):
        for m in ("tfidf", "bm25", "embedding", "hybrid"):
            S = ENGINE.matrix(m)
            self.assertEqual(S.shape, (32, 32)); self.assertTrue(np.allclose(np.diag(S), 0))
            self.assertTrue(S.min() >= -1e-9 and S.max() <= 1 + 1e-9)

    def test_symmetry(self):
        for m in ("tfidf", "bm25", "embedding", "hybrid"):
            S = ENGINE.matrix(m); self.assertTrue(np.allclose(S, S.T, atol=1e-8))

    def test_identical_profiles_are_most_similar(self):
        docs = ["knowledge graph ontology", "knowledge graph ontology", "intrusion malware security"]
        S, _, _ = sim.tfidf_matrix(docs); self.assertGreater(S[0, 1], 0.99); self.assertLess(S[0, 2], 0.1)

    def test_hybrid_alpha_extremes(self):
        self.assertTrue(np.allclose(ENGINE.matrix("hybrid", 1.0), ENGINE.matrix("embedding")))
        self.assertTrue(np.allclose(ENGINE.matrix("hybrid", 0.0), ENGINE.matrix("tfidf")))


class TestEngine(unittest.TestCase):
    def test_output_schema(self):
        out = ENGINE.find_related("F05", k=3)
        self.assertEqual(out["input_id"], "F05"); res = out["result"]
        for key in ("researcher", "top_keywords", "entities", "related_researchers", "common_topics"):
            self.assertIn(key, res)
        self.assertEqual(len(res["related_researchers"]), 3)
        sims = [r["similarity"] for r in res["related_researchers"]]
        self.assertEqual(sims, sorted(sims, reverse=True)); self.assertNotIn("F05", [r["faculty_id"] for r in res["related_researchers"]])
        json.dumps(out)  # must be JSON serialisable

    def test_related_is_topically_sensible(self):
        names = {r["faculty_id"] for r in ENGINE.find_related("F05", k=3, method="tfidf")["result"]["related_researchers"]}
        self.assertTrue(names <= {f"F{i:02d}" for i in range(5, 9)} | {"F01", "F02", "F03", "F04"}, names)

    def test_unknown_and_bad_method(self):
        with self.assertRaises(KeyError): ENGINE.find_related("NOPE")
        with self.assertRaises(ValueError): ENGINE.find_related("F01", method="magic")

    def test_search_prefers_security_for_security_query(self):
        top = ENGINE.search("network intrusion detection and malware", k=3, method="tfidf")["results"]
        self.assertIn(top[0]["faculty_id"], {f"F{i}" for i in range(21, 25)})

    def test_topics_and_kg(self):
        self.assertGreaterEqual(ENGINE.topics.k, 6)
        G = ENGINE.knowledge_graph(); rels = {d["rel"] for _, _, d in G.edges(data=True)}
        self.assertTrue({"PUBLISHED", "WORKS_ON", "USES", "RELATED_TO", "COLLABORATION_CANDIDATE"} <= rels)
        self.assertEqual(sum(1 for _, d in G.nodes(data=True) if d["type"] == "Faculty"), 32)
        self.assertTrue(ENGINE.kg_facts("F05"))

    def test_reproducible(self):
        e2 = ResearchDiscoveryEngine(_DF, embedder=Embedder("lsa"))
        self.assertTrue(np.allclose(e2.matrix("hybrid"), ENGINE.matrix("hybrid")))


class TestEntities(unittest.TestCase):
    def test_lexicon(self):
        e = extract_entities("We fine-tune BERT for named entity recognition on a knowledge graph with a U-Net.")
        self.assertIn("bert", e["methods"]); self.assertIn("u-net", e["methods"])
        self.assertIn("named entity recognition", e["tasks"]); self.assertIn("knowledge graph", e["domains"])

    def test_no_partial_word_match(self):
        self.assertEqual(extract_entities("The ragged cnnection")["methods"], [])

    def test_triples(self):
        t = extract_triples("P1", "Entity linking with BERT"); self.assertIn(("P1", "USES", "bert"), t)


class TestMetrics(unittest.TestCase):
    def test_hand_computed(self):
        hits = np.array([0, 1, 0, 1, 0])
        self.assertAlmostEqual(M.precision_at_k(hits, 3), 1 / 3); self.assertAlmostEqual(M.recall_at_k(hits, 2, 3), 0.5)
        self.assertAlmostEqual(M.reciprocal_rank(hits), 0.5); self.assertAlmostEqual(M.average_precision(hits), (1 / 2 + 2 / 4) / 2)

    def test_ndcg_perfect_and_worst(self):
        self.assertAlmostEqual(M.ndcg_at_k([2, 1, 0], [2, 1, 0], 3), 1.0)
        self.assertLess(M.ndcg_at_k([0, 1, 2], [2, 1, 0], 3), 1.0)

    def test_perfect_ranker(self):
        REL = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]]); S = REL.astype(float)
        pq, kept = M.per_query_metrics(S, REL, [0, 1, 2]); self.assertEqual(kept, [0, 1])
        self.assertEqual(pq["P@1"].mean(), 1.0); self.assertEqual(pq["MRR"].mean(), 1.0)

    def test_kappa(self):
        self.assertAlmostEqual(cohens_kappa([0, 1, 2, 1], [0, 1, 2, 1]), 1.0)
        self.assertLess(cohens_kappa([0, 1, 2, 1], [1, 0, 1, 2]), 0.1)

    def test_pairs_loader(self):
        R = from_pairs(ENGINE.ids, SAMPLE_PAIRS); self.assertTrue((R == R.T).all()); self.assertEqual(R.shape, (32, 32))


class TestHandlers(unittest.TestCase):
    def test_validation_errors(self):
        for fn, body, status in [(handlers.find_related_researchers, {}, 422), (handlers.find_related_researchers, {"faculty_id": "ZZ"}, 404),
                                 (handlers.find_related_researchers, {"faculty_id": "F01", "k": 0}, 422),
                                 (handlers.find_related_researchers, {"faculty_id": "F01", "method": "x"}, 422),
                                 (handlers.search_researchers, {"text": "  "}, 422), (handlers.extract_entities, {"text": 5}, 422)]:
            with self.assertRaises(handlers.ApiError) as cm: fn(body)
            self.assertEqual(cm.exception.status, status)


class TestHttpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(0); cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def _post(self, path, payload):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    def test_health_and_related(self):
        self.assertEqual(json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health"))["status"], "ok")
        self.assertEqual(len(self._post("/find-related-researchers", {"faculty_id": "F01", "k": 2})["result"]["related_researchers"]), 2)

    def test_error_codes(self):
        with self.assertRaises(urllib.error.HTTPError) as cm: self._post("/find-related-researchers", {"faculty_id": "nope"})
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as cm: self._post("/nothing", {})
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
