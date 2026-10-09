"""Interactive demo.   streamlit run demo/streamlit_app.py     (untested in the build sandbox: Streamlit not installed there)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from src.models.engine import load_engine, METHODS

st.set_page_config(page_title="Research Discovery Engine", layout="wide")


@st.cache_resource
def engine():
    return load_engine()


e = engine()
st.title("Research Discovery & Collaboration Engine")
st.caption("AI University platform - Group 7 NLP asset")
if e.data_source != "REAL":
    st.warning(f"Running on **{e.data_source}** data. Results are for demonstration only.")
if e.embedder.backend != "sentence-transformer":
    st.info("Transformer model not installed: using the LSA fallback for the 'embedding' method.")

tab1, tab2, tab3, tab4 = st.tabs(["Find related researchers", "Search by interests", "Topics & keywords", "Knowledge graph"])

with tab1:
    names = {f"{f['faculty_name']} ({f['faculty_id']}) - {f['department']}": f["faculty_id"] for f in e.list_faculty()}
    c1, c2, c3 = st.columns([3, 1, 1])
    fid = names[c1.selectbox("Researcher", list(names))]
    k = c2.slider("Top-K", 1, 10, 3); method = c3.selectbox("Method", METHODS, index=3)
    out = e.find_related(fid, k, method)["result"]
    st.subheader(f"Related to {out['researcher']}")
    for r in out["related_researchers"]:
        st.markdown(f"**{r['faculty_name']}** ({r['faculty_id']}, {r['department']}) - similarity **{r['similarity']}**")
        st.caption("Shared keywords: " + (", ".join(r["shared_keywords"]) or "-") + " | Shared topics: " + (" ; ".join(t["label"] for t in r["shared_topics"]) or "-"))
    st.markdown("**Common topics:** " + (", ".join(out["common_topics"]) or "-"))
    st.markdown("**Methods / tasks detected:** " + ", ".join(out["entities"]["methods"] + out["entities"]["tasks"]))
    st.caption(e.find_related(fid, k, method)["disclaimer"])

with tab2:
    q = st.text_area("Describe your interests", "I want to detect prompt injection attacks on LLM chatbots")
    if st.button("Search"):
        st.table(pd.DataFrame(e.search(q, 5)["results"]))

with tab3:
    st.json(e.topic_summary()["topics"])
    txt = st.text_area("Extract keywords / entities from text", "We fine-tune BERT for entity linking over a knowledge graph.")
    st.write(e.extract_keywords(txt)); st.json(e.extract_entities(txt))

with tab4:
    import matplotlib; matplotlib.use("Agg")
    from src.models.kg import draw_ego
    import tempfile
    f2 = names[st.selectbox("Ego graph for", list(names), key="kg")]
    p = os.path.join(tempfile.gettempdir(), f"kg_{f2}.png"); draw_ego(e.knowledge_graph(), f2, p); st.image(p)
    st.write(pd.DataFrame(e.kg_facts(f2), columns=["subject", "relation", "object"]))
