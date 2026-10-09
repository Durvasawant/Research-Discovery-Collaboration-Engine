# Annotation Guidelines - researcher-pair relevance

Two team members label **independently** (do not discuss until both finish). For each pair, read both researchers' interests and paper titles, then assign:

| Label | Meaning | Test question |
|---|---|---|
| **2** | closely related | "Would these two plausibly co-author a paper or supervise a joint student *tomorrow*?" |
| **1** | partially related | "Do they share a method, data type or sub-topic, so an introduction would be useful?" |
| **0** | unrelated | Only generic overlap ("both use machine learning", "both write Python"). |

Rules: judge **research content**, not seniority, department or publication count. Generic vocabulary (deep learning, neural networks, benchmark, dataset) never makes a pair related by itself. When unsure choose the lower label and add a note.
After labelling run `scripts/make_annotation_sheet.py --merge`: report raw agreement and **Cohen's kappa** (target >= 0.6), adjudicate disagreements together, record how many there were.
Pooling note: only pairs proposed by at least one method (plus random negatives) are labelled; unlabelled pairs count as 0. State this in the report.
