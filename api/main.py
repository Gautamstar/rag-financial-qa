import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.pipeline import stream_query, query as rag_query, retrieve

app = FastAPI(title="Financial RAG Q&A")


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] | None = None
    company_filter: list[str] | None = None
    temperature: float = 0.0
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    sources: list[str]


class MetricsResponse(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None
    result = rag_query(
        req.question,
        chat_history=history,
        company_filter=req.company_filter,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return QueryResponse(**result)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None
    return StreamingResponse(
        stream_query(
            req.question,
            chat_history=history,
            company_filter=req.company_filter,
            temperature=req.temperature,
            top_k=req.top_k,
        ),
        media_type="text/plain",
    )


@app.post("/retrieve", response_model=QueryResponse)
def retrieve_sources(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = retrieve(req.question, company_filter=req.company_filter, top_k=req.top_k)
    return QueryResponse(**result)


@app.post("/metrics", response_model=MetricsResponse)
def run_metrics():
    """Run RAGAS evaluation over the built-in eval set. Takes several minutes."""
    from src.evaluate import run_evaluation
    result = run_evaluation()
    df = result.to_pandas()
    scores = {}
    if "faithfulness" in df.columns:
        scores["faithfulness"] = round(float(df["faithfulness"].mean()), 4)
    if "answer_relevancy" in df.columns:
        scores["answer_relevancy"] = round(float(df["answer_relevancy"].mean()), 4)
    return MetricsResponse(**scores)
