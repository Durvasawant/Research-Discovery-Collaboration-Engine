"""Small Research Knowledge Graph (stretch goal), built with networkx.

Node types : Faculty, Paper, Topic, Method, Task
Edge types : PUBLISHED (Faculty->Paper), WORKS_ON (Faculty->Topic), RELATED_TO (Paper->Topic),
             USES (Paper->Method), ADDRESSES (Paper->Task), COLLABORATION_CANDIDATE (Faculty<->Faculty)
"""
import json
import networkx as nx
from src.models.entities import extract_triples

COLORS = {"Faculty": "#4C78A8", "Paper": "#F58518", "Topic": "#54A24B", "Method": "#B279A2", "Task": "#E45756"}


def build_graph(researchers, topic_model, paper_topic_weights, faculty_topic_weights, S_rel,
                topic_min=0.20, collab_threshold=None, collab_top=2):
    G = nx.MultiDiGraph()
    ids = [r.faculty_id for r in researchers]
    for t in range(topic_model.k):
        G.add_node(f"T{t}", type="Topic", label=topic_model.label(t))
    pi = 0
    for ri, r in enumerate(researchers):
        G.add_node(r.faculty_id, type="Faculty", label=r.name, department=r.department)
        for t in faculty_topic_weights[ri].argsort()[::-1][:3]:
            if faculty_topic_weights[ri][t] >= topic_min:
                G.add_edge(r.faculty_id, f"T{t}", rel="WORKS_ON", weight=round(float(faculty_topic_weights[ri][t]), 3))
        for p in r.papers:
            pid = f"{r.faculty_id}-P{pi}"; pi += 1
            G.add_node(pid, type="Paper", label=p.title)
            G.add_edge(r.faculty_id, pid, rel="PUBLISHED")
            w = paper_topic_weights[pid]
            t = int(w.argmax())
            G.add_edge(pid, f"T{t}", rel="RELATED_TO", weight=round(float(w[t]), 3))
            for s, pred, o in extract_triples(pid, p.text):
                if pred == "IN_DOMAIN":
                    continue
                ntype = "Method" if pred == "USES" else "Task"
                G.add_node(f"{ntype}:{o}", type=ntype, label=o)
                G.add_edge(pid, f"{ntype}:{o}", rel=pred)
    # collaboration candidates: top-N most similar per faculty (mutual edges are added once)
    added = set()
    for i, fid in enumerate(ids):
        order = S_rel[i].argsort()[::-1]
        order = [j for j in order if j != i][:collab_top]
        for j in order:
            if collab_threshold is not None and S_rel[i, j] < collab_threshold:
                continue
            key = tuple(sorted((ids[i], ids[j])))
            if key not in added:
                added.add(key)
                G.add_edge(ids[i], ids[j], rel="COLLABORATION_CANDIDATE", weight=round(float(S_rel[i, j]), 3))
    return G


def to_json(G):
    return {"nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]}


def ego_facts(G, faculty_id, max_papers=5):
    """Human-readable triples around one faculty member."""
    facts = []
    for _, v, d in G.out_edges(faculty_id, data=True):
        if d["rel"] == "WORKS_ON":
            facts.append((G.nodes[faculty_id]["label"], "WORKS_ON", G.nodes[v]["label"]))
        elif d["rel"] == "PUBLISHED":
            facts.append((G.nodes[faculty_id]["label"], "PUBLISHED", G.nodes[v]["label"]))
            for _, w, d2 in G.out_edges(v, data=True):
                facts.append((G.nodes[v]["label"], d2["rel"], G.nodes[w]["label"]))
    return facts


def draw_ego(G, faculty_id, path, radius=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = nx.ego_graph(G.to_undirected(as_view=False), faculty_id, radius=radius)
    # keep the figure readable: drop other faculty's papers
    keep = [n for n in H if G.nodes[n]["type"] != "Paper" or n.startswith(faculty_id + "-")]
    H = H.subgraph(keep)
    pos = nx.spring_layout(H, seed=7, k=0.9)
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(H, pos, alpha=0.35, ax=ax)
    for typ, col in COLORS.items():
        nodes = [n for n in H if G.nodes[n]["type"] == typ]
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_color=col, node_size=650 if typ == "Faculty" else 320, label=typ, ax=ax)
    labels = {n: (G.nodes[n]["label"][:34] + ("..." if len(G.nodes[n]["label"]) > 34 else "")) for n in H}
    nx.draw_networkx_labels(H, pos, labels, font_size=7, ax=ax)
    ax.legend(loc="lower left", fontsize=8); ax.axis("off")
    ax.set_title(f"Research knowledge graph around {G.nodes[faculty_id]['label']} ({faculty_id})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
