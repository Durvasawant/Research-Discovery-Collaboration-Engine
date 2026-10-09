# Integration with the AI University Platform

| Consumer | Direction | What flows | Endpoint | Status |
|---|---|---|---|---|
| **Group 8 - Placement & Career Intelligence** | G7 -> G8 | research skills/methods of faculty (`entities`, `top_keywords`) to seed the skill ontology; reverse: job-description text -> faculty with matching expertise (mentor/guide finder) | `/extract-entities`, `/search-researchers` | demo script `demo/client_example.py` |
| **Group 4 - University Knowledge Assistant** | G7 -> G4 | faculty profile cards as retrievable documents ("Who works on knowledge graphs?") | `/find-related-researchers`, `/faculty` | schema agreed in this doc |
| **Group 9 - Curriculum Intelligence** | G9 -> G7 | course-outcome text -> topics; map faculty expertise to courses | `/extract-keywords`, `/search-researchers` | proposed |
| **Group 5 - Feedback Intelligence** | G7 -> G5 | the same embedding/keyword service can be reused for aspect clustering | python module | proposed |

**Shared contract** (all groups): requests/responses are JSON objects with `input_id`, `result`, optional `confidence`/`score`; see `docs/API.md`. Start-up for a peer: `python -m src.api.simple_server --port 8000`, set `RDCE_URL` and run `demo/client_example.py`.
**To complete with partner groups (Week 7):** run one live call with their component and record the exchanged JSON in this file.
