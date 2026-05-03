import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.pipeline import stream_query, query as rag_query, retrieve

app = FastAPI(title="Financial RAG Q&A")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag_query(req.question)
    return QueryResponse(**result)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return StreamingResponse(stream_query(req.question), media_type="text/plain")


@app.post("/retrieve", response_model=QueryResponse)
def retrieve_sources(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = retrieve(req.question)
    return QueryResponse(**result)
