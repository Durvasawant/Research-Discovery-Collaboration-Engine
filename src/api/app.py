"""FastAPI application.   uvicorn src.api.app:app --reload     (docs at http://127.0.0.1:8000/docs)

NOTE: thin wrapper over src/api/handlers.py -- all logic lives there."""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.api import handlers

app = FastAPI(title="Research Discovery & Collaboration Engine", version="1.0.0",
              description="Group 7 NLP asset of the AI University platform: related-researcher discovery, keywords, NER, topics, knowledge graph.")


class RelatedRequest(BaseModel):
    faculty_id: str = Field(..., examples=["F05"])
    k: int = Field(3, ge=1, le=20)
    method: str = Field("hybrid", description="tfidf | bm25 | embedding | hybrid")


class SearchRequest(BaseModel):
    text: str = Field(..., examples=["detecting prompt injection in LLM chatbots"])
    k: int = Field(5, ge=1, le=20)
    method: str = Field("hybrid", description="tfidf | embedding | hybrid")


class TextRequest(BaseModel):
    text: str
    input_id: Optional[str] = None
    n: int = Field(8, ge=1, le=30)


def _run(fn, body):
    try:
        return fn(body)
    except handlers.ApiError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@app.get("/health")
def health(): return handlers.health()
@app.get("/faculty")
def faculty(): return handlers.list_faculty()
@app.get("/topics")
def topics(): return handlers.topics()
@app.get("/knowledge-graph")
def kg(): return handlers.knowledge_graph()
@app.post("/find-related-researchers")
def related(req: RelatedRequest): return _run(handlers.find_related_researchers, req.model_dump())
@app.post("/search-researchers")
def search(req: SearchRequest): return _run(handlers.search_researchers, req.model_dump())
@app.post("/extract-keywords")
def keywords(req: TextRequest): return _run(handlers.extract_keywords, req.model_dump())
@app.post("/extract-entities")
def entities(req: TextRequest): return _run(handlers.extract_entities, req.model_dump())
