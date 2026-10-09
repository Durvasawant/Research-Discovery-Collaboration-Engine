"""Generate a SYNTHETIC faculty/paper dataset used only for development and tests.

Everything here is fictional. Results computed on this data are NOT reportable
as final results -- replace data/faculty_research.csv with real, collected data.

Outputs
  data/sample_faculty_research.csv   one row per paper (faculty metadata repeated)
  data/sample_relevance_pairs.csv    graded pair labels (2 = same primary area, 1 = related via a secondary area, 0 = unrelated)
"""
import itertools, os, random
import pandas as pd

SEED = 42
random.seed(SEED)

# (title, abstract, keywords, area-tags)   first tag = primary area of the paper
PAPERS = {
"NLP": [
 ("Transformer-based Named Entity Recognition for Low-Resource Domains","We fine-tune BERT for named entity recognition in low-resource domains and study how annotation effort affects tagging accuracy.","NER; BERT; transformers; low-resource",["NLP"]),
 ("Relation Extraction from Scientific Text","We propose a neural pipeline that extracts relations between methods and tasks mentioned in scientific articles.","information extraction; relation extraction; scientific text",["NLP"]),
 ("Aspect-Based Sentiment Analysis of Student Feedback","We classify sentiment towards teaching aspects in free-text course feedback using sentence embeddings and a linear classifier.","sentiment analysis; aspect extraction; text classification",["NLP"]),
 ("Cross-Lingual Transfer for Indian Language Text Classification","We study multilingual transformers for text classification in Hindi and Marathi with very little labelled data.","multilingual NLP; transfer learning; Hindi; text classification",["NLP"]),
 ("Abstractive Summarisation of Long Documents","We evaluate encoder-decoder transformers for summarising long documents and measure factual consistency with the source.","summarisation; transformers; factual consistency",["NLP"]),
 ("Sentence Embeddings for Semantic Textual Similarity","We compare contrastively trained sentence embeddings with TF-IDF baselines on semantic textual similarity benchmarks.","sentence embeddings; semantic similarity; TF-IDF",["NLP"]),
],
"KG": [
 ("Entity Linking for Knowledge Graph Population","We link extracted mentions to knowledge graph entities using contextual candidate ranking.","entity linking; knowledge graph; ranking",["KG"]),
 ("Ontology-driven Semantic Search over Academic Data","We design an ontology and a SPARQL-based semantic search system for university records.","ontology; semantic search; SPARQL",["KG"]),
 ("Graph Embeddings for Link Prediction in Knowledge Graphs","We evaluate translation-based graph embeddings for link prediction on large knowledge graphs.","knowledge graph; graph embeddings; link prediction",["KG"]),
 ("Automatic Knowledge Graph Construction from Text","A pipeline combining named entity recognition and relation extraction builds a knowledge graph from raw documents.","knowledge graph; information extraction; NER",["KG","NLP"]),
 ("Linked Data Integration for Research Information Systems","We integrate heterogeneous publication records into a linked-data repository using shared vocabularies.","linked data; RDF; research information systems",["KG"]),
 ("Reasoning over Ontologies with Description Logics","We study scalable reasoning over OWL ontologies and its use in data validation.","ontology; OWL; reasoning",["KG"]),
],
"LLM": [
 ("Retrieval-Augmented Generation for Question Answering","We combine dense retrieval with large language models to answer domain questions with source citations.","RAG; LLM; dense retrieval; question answering",["LLM","NLP"]),
 ("Reducing Hallucination in Generative AI Systems","We study prompting and grounding strategies that reduce hallucination in large language models.","hallucination; generative AI; LLM; grounding",["LLM"]),
 ("Parameter-Efficient Fine-Tuning of Large Language Models","We compare LoRA and adapter methods for adapting large language models to specialised domains on a small compute budget.","LoRA; fine-tuning; large language models",["LLM"]),
 ("Evaluating LLM-as-a-Judge for Essay Scoring","We examine agreement between large language model judges and human graders when scoring student essays against rubrics.","LLM-as-a-judge; automated scoring; evaluation",["LLM","NLP"]),
 ("Instruction Tuning and Alignment of Conversational Agents","We analyse instruction tuning data and preference optimisation for aligning conversational language models.","instruction tuning; alignment; chatbots",["LLM"]),
 ("Prompt Engineering Strategies for Reasoning Tasks","We benchmark chain-of-thought and self-consistency prompting for multi-step reasoning with generative AI models.","prompting; chain-of-thought; reasoning; generative AI",["LLM"]),
],
"CV": [
 ("U-Net Variants for Tumour Segmentation in MRI","We compare convolutional segmentation networks for brain tumour segmentation in MRI scans.","segmentation; U-Net; MRI; deep learning",["CV","HEALTH"]),
 ("Object Detection with Lightweight Convolutional Networks","We propose a lightweight CNN detector for real-time object detection on embedded devices.","object detection; CNN; real-time",["CV"]),
 ("Multi-Object Tracking in Surveillance Video","We present a tracking-by-detection method with appearance embeddings for crowded video scenes.","object tracking; video analytics; re-identification",["CV"]),
 ("Action Recognition using 3D Convolutional Networks","A 3D CNN architecture is trained for human action recognition in videos.","action recognition; 3D CNN; video",["CV"]),
 ("Vision-Language Pretraining for Image Captioning","We pretrain a transformer on paired images and text to generate image captions.","vision-language; transformers; image captioning",["CV","NLP"]),
 ("Self-Supervised Representation Learning for Images","We study contrastive self-supervised pretraining of visual encoders with limited labels.","self-supervised learning; contrastive learning; image representation",["CV"]),
],
"HEALTH": [
 ("Deep Learning for Chest X-ray Diagnosis","We train convolutional networks to detect pneumonia from chest X-ray images.","chest X-ray; diagnosis; deep learning; CNN",["HEALTH","CV"]),
 ("Predicting Diabetes Risk from Patient Records","We compare gradient boosting and logistic regression for disease risk prediction on patient records.","disease prediction; machine learning; healthcare",["HEALTH"]),
 ("Clinical Named Entity Recognition in Electronic Health Records","We adapt transformer models to extract diseases and medications from clinical notes.","clinical NER; EHR; transformers; biomedical NLP",["HEALTH","NLP"]),
 ("Summarising Discharge Summaries with Language Models","We evaluate abstractive summarisation of hospital discharge summaries with factual consistency checks.","summarisation; clinical text; factual consistency",["HEALTH","NLP"]),
 ("Federated Learning for Privacy-Preserving Hospital Analytics","We train predictive models across hospitals without sharing patient data using federated learning.","federated learning; privacy; healthcare analytics",["HEALTH"]),
 ("Wearable Sensor Analytics for Cardiac Monitoring","We detect arrhythmias from wearable ECG sensors using lightweight classifiers.","wearables; ECG; arrhythmia detection",["HEALTH","IOT"]),
],
"SEC": [
 ("Anomaly-based Network Intrusion Detection using Autoencoders","We detect network intrusions with autoencoders trained on benign traffic.","intrusion detection; anomaly detection; autoencoder",["SEC"]),
 ("Adversarial Attacks on Malware Classifiers","We analyse how adversarial perturbations evade machine-learning malware classifiers.","adversarial attacks; malware; cybersecurity",["SEC"]),
 ("Prompt Injection Attacks on LLM Applications","We characterise prompt injection attacks against large language model applications and evaluate detection defences.","prompt injection; LLM security; attack detection",["SEC","LLM"]),
 ("Lightweight Authentication for IoT Edge Devices","We propose a lightweight authentication protocol for resource-constrained IoT devices.","IoT; authentication; edge computing; security",["SEC","IOT"]),
 ("Phishing Email Detection with Text Classification","We detect phishing emails using lexical features and transformer classifiers.","phishing; email security; text classification",["SEC","NLP"]),
 ("Blockchain-based Access Control for Sensitive Records","We design smart-contract access control for sharing sensitive institutional records.","blockchain; access control; privacy",["SEC"]),
],
"IOT": [
 ("Energy-Efficient Routing in Wireless Sensor Networks","We design an energy-aware routing protocol that extends wireless sensor network lifetime.","wireless sensor networks; routing; energy efficiency",["IOT"]),
 ("Smart Campus Energy Monitoring using IoT Sensors","We deploy IoT sensors across a campus and forecast building energy consumption.","smart campus; IoT; energy; time series",["IOT","DATA"]),
 ("LSTM-based Forecasting of Sensor Time Series","We compare LSTM and ARIMA models for forecasting multivariate sensor time series.","LSTM; time series forecasting; sensors",["IOT","DATA"]),
 ("Edge Computing for Real-Time Industrial Analytics","We offload stream analytics to edge nodes to reduce latency in industrial monitoring.","edge computing; stream processing; latency",["IOT"]),
 ("LoRaWAN Deployment for Agricultural Monitoring","We evaluate long-range low-power wireless links for soil and weather sensing on farms.","LoRaWAN; agriculture; low-power networks",["IOT"]),
 ("Digital Twins for Smart Building Management","We build digital twins that combine sensor streams and simulation to optimise building operation.","digital twin; smart buildings; sensors",["IOT"]),
],
"DATA": [
 ("Time Series Anomaly Detection in Financial Data","We evaluate statistical and neural detectors for anomalies in transaction time series.","anomaly detection; time series; finance",["DATA"]),
 ("Scalable Frequent Pattern Mining on Big Data","We parallelise frequent itemset mining on distributed platforms for large transaction logs.","data mining; frequent patterns; big data",["DATA"]),
 ("Learning Analytics for Predicting Student Dropout","We predict student dropout from learning-management-system activity logs using tree ensembles.","learning analytics; dropout prediction; education data",["DATA"]),
 ("Recommender Systems with Implicit Feedback","We compare matrix factorisation and neural recommenders on implicit feedback datasets.","recommender systems; collaborative filtering; matrix factorisation",["DATA"]),
 ("Graph Clustering for Community Detection in Collaboration Networks","We cluster co-authorship graphs to detect research communities and measure cluster quality.","community detection; graph clustering; co-authorship",["DATA","KG"]),
 ("Explainable Machine Learning for Tabular Data","We study SHAP-based explanations for gradient boosted models on tabular benchmarks.","explainability; SHAP; tabular data",["DATA"]),
],
}

INTERESTS = {
 "NLP":["natural language processing, information extraction, named entity recognition","text classification, sentiment analysis, multilingual NLP","summarisation, semantic similarity, sentence embeddings"],
 "KG":["knowledge graphs, entity linking, semantic web","ontology engineering, knowledge graph construction, semantic search","linked data, RDF, reasoning over ontologies"],
 "LLM":["large language models, generative AI, retrieval augmented generation","LLM alignment, prompt engineering, hallucination","fine-tuning language models, LLM evaluation"],
 "CV":["image segmentation, object detection, deep learning","video analytics, object tracking, convolutional neural networks","self-supervised vision, multimodal learning"],
 "HEALTH":["medical imaging diagnosis, disease prediction, machine learning in healthcare","clinical text mining, electronic health records, biomedical NLP","privacy-preserving healthcare analytics, wearables"],
 "SEC":["network intrusion detection, cybersecurity, anomaly detection","adversarial machine learning, malware, LLM security","authentication, access control, secure systems"],
 "IOT":["internet of things, edge computing, wireless sensor networks","smart campus sensing, IoT analytics, time series forecasting","digital twins, smart buildings, low-power networks"],
 "DATA":["data mining, time series analysis, big data","learning analytics, educational data mining, recommender systems","graph analytics, community detection, explainable machine learning"],
}
DEPT = {"NLP":"Computer Science","KG":"Computer Science","LLM":"Computer Science","CV":"Electronics","HEALTH":"Health Informatics","SEC":"Information Technology","IOT":"Information Technology","DATA":"Data Science"}
SECONDARY = {"NLP":"KG","KG":"NLP","LLM":"SEC","CV":"HEALTH","HEALTH":"NLP","SEC":"LLM","IOT":"DATA","DATA":"IOT"}

import re
# --- difficulty injection -------------------------------------------------------------------------------
# k==3 -> PARAPHRASE faculty: domain vocabulary replaced by descriptive paraphrases (little lexical overlap with peers)
# k==2 -> TRAP faculty: generic ML boilerplate appended (lexical overlap with unrelated areas)
PARA = {"knowledge graphs":"networks of linked facts","knowledge graph":"network of linked facts","ontologies":"formal concept schemas",
 "ontology":"formal concept schema","named entity recognition":"finding names of people and organisations in text","entity linking":"matching mentions to catalogue entries",
 "large language models":"generative neural text engines","large language model":"generative neural text engine","llms":"generative text engines","llm":"generative text engine",
 "sentiment analysis":"opinion mining","sentiment":"opinion","classification":"categorisation","classifiers":"categorisers","detection":"discovery",
 "detect":"discover","image":"picture","images":"pictures","segmentation":"region partitioning","wireless sensor networks":"distributed low-power sensing nodes",
 "intrusion":"unauthorised access","attacks":"abuse attempts","attack":"abuse attempt","patient":"care recipient","hospital":"care centre","forecasting":"predicting future values",
 "summarisation":"condensing","summarising":"condensing","retrieval":"look-up","embeddings":"vector representations","embedding":"vector representation","malware":"hostile software",
 "transformer":"attention network","transformers":"attention networks","cnn":"convolutional network","convolutional networks":"layered image filters","sensors":"measurement devices",
 "sensor":"measurement device","iot":"connected devices","security":"protection","cybersecurity":"digital protection","privacy":"confidentiality","clinical":"bedside",
 "healthcare":"medical care","time series":"temporal measurements","data mining":"pattern discovery","recommender systems":"suggestion engines","tracking":"following","video":"moving pictures"}
_PARA_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in sorted(PARA, key=len, reverse=True)) + r")(?![\w-])", re.I)
def paraphrase(text): return _PARA_RE.sub(lambda m: PARA[m.group(0).lower()], text)
BOILER = ["We train deep neural networks with learned embeddings on public benchmark datasets and report accuracy, precision and recall.",
          "The proposed framework uses attention and representation learning, and experiments show consistent improvements over strong baselines.",
          "Evaluation covers several datasets, an ablation study and a discussion of scalability and deployment."]
BOILER_KW = "deep learning; embeddings; neural networks; benchmark"

# 4 faculty per primary area = 32 faculty, 3 papers each (2 from primary pool + 1 from primary or secondary pool)
def build():
    rows, fac_areas = [], {}
    n = 0
    for area, pool in PAPERS.items():
        idx = list(range(len(pool)))
        for k in range(4):
            n += 1
            fid = f"F{n:02d}"
            random.shuffle(idx)
            chosen = [pool[i] for i in idx[:2]]
            if k % 2 == 1:   # half of the faculty also publish in a secondary area -> realistic overlap
                sec = SECONDARY[area]
                chosen.append(random.choice(PAPERS[sec]))
            else:
                chosen.append(pool[idx[2]])
            interest = INTERESTS[area][k % 3]
            if k == 3:
                chosen = [(paraphrase(t), paraphrase(a), paraphrase(kw), tg) for t, a, kw, tg in chosen]
                interest = paraphrase(interest)
            elif k == 2:
                chosen = [(t, a + " " + " ".join(random.sample(BOILER, 2)), kw + "; " + BOILER_KW, tg) for t, a, kw, tg in chosen]
            tags = {area}
            for t, a, kw, tg in chosen: tags |= set(tg)
            fac_areas[fid] = (area, tags)
            for t, a, kw, tg in chosen:
                rows.append(dict(faculty_id=fid, faculty_name=f"Faculty {fid[1:]}", department=DEPT[area],
                                 research_interests=interest, paper_title=t, abstract=a, keywords=kw,
                                 areas=";".join(sorted(tags))))
    return pd.DataFrame(rows), fac_areas

if __name__ == "__main__":
    df, fa = build()
    out = os.path.join(os.path.dirname(__file__), "..", "data")
    df.to_csv(os.path.join(out, "sample_faculty_research.csv"), index=False)
    pairs = []
    for a, b in itertools.combinations(sorted(fa), 2):
        pa, ta = fa[a]; pb, tb = fa[b]
        lab = 2 if pa == pb else (1 if (ta & tb) else 0)
        pairs.append(dict(faculty_id_a=a, faculty_id_b=b, label=lab))
    pd.DataFrame(pairs).to_csv(os.path.join(out, "sample_relevance_pairs.csv"), index=False)
    print(len(df), "paper rows;", len(fa), "faculty;", len(pairs), "pairs;",
          pd.DataFrame(pairs).label.value_counts().to_dict())
