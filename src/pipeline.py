import hashlib
import json
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

VECTORSTORE_DIR = Path("vectorstore")
BM25_CACHE = VECTORSTORE_DIR / "bm25.pkl"
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")
TOP_K = 4
RRF_K = 60

COMPANY_ALIASES = {
    "jpmorgan": "jpm",
    "jp morgan": "jpm",
    "jpmorgan chase": "jpm",
    "goldman sachs": "gs",
    "goldman": "gs",
    "metlife": "met",
    "prudential": "pru",
    "prudential financial": "pru",
    "aig": "aig",
    "american international group": "aig",
}

_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""Use the following excerpts from financial documents to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}
{chat_history}
Question: {question}

Answer:""",
)

_faiss = None
_bm25 = None
_bm25_docs = None
_llm_chain = None


def _normalize_query(question: str) -> str:
    q = question.lower()
    for name, ticker in COMPANY_ALIASES.items():
        if name in q:
            q = q + " " + ticker
    return q


def _doc_key(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def _format_chat_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = ["\nPrevious conversation:"]
    for msg in history:
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines)


def _filter_by_company(docs: list[Document], company_filter: list[str] | None) -> list[Document]:
    if not company_filter:
        return docs
    tickers = {c.upper() for c in company_filter}
    filtered = [d for d in docs if d.metadata.get("ticker", "").upper() in tickers]
    return filtered if filtered else docs


def _load():
    global _faiss, _bm25, _bm25_docs, _llm_chain
    if _llm_chain is not None:
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _faiss = FAISS.load_local(
        str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
    )

    with open(VECTORSTORE_DIR / "chunks.json") as f:
        raw = json.load(f)
    _bm25_docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in raw]

    if BM25_CACHE.exists():
        with open(BM25_CACHE, "rb") as f:
            _bm25 = pickle.load(f)
    else:
        tokenized = [doc.page_content.lower().split() for doc in _bm25_docs]
        _bm25 = BM25Okapi(tokenized)
        with open(BM25_CACHE, "wb") as f:
            pickle.dump(_bm25, f)

    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0,
    )
    _llm_chain = _PROMPT | llm | StrOutputParser()


def _hybrid_retrieve(
    question: str,
    k: int = TOP_K,
    company_filter: list[str] | None = None,
) -> list[Document]:
    n_candidates = k * 10
    normalized = _normalize_query(question)

    faiss_results = _faiss.similarity_search(question, k=n_candidates)
    for doc in faiss_results:
        doc.metadata["_retriever"] = "FAISS"

    tokens = normalized.lower().split()
    bm25_scores = _bm25.get_scores(tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    bm25_results = [_bm25_docs[i] for i in bm25_top_idx]
    for doc in bm25_results:
        doc.metadata["_retriever"] = "BM25"

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(faiss_results):
        key = _doc_key(doc)
        if key not in rrf_scores:
            doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)

    for rank, doc in enumerate(bm25_results):
        key = _doc_key(doc)
        if key in rrf_scores:
            doc_map[key].metadata["_retriever"] = "FAISS+BM25"
        else:
            doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_docs = [doc_map[key] for key in ranked]

    filtered = _filter_by_company(top_docs, company_filter)
    return filtered[:k]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _source_label(doc: Document) -> str:
    source = doc.metadata.get("source", doc.metadata.get("filename", ""))
    retriever = doc.metadata.get("_retriever", "")
    return f"{source} [{retriever}]" if retriever else source


def retrieve(
    question: str,
    company_filter: list[str] | None = None,
) -> dict:
    _load()
    docs = _hybrid_retrieve(question, company_filter=company_filter)
    return {
        "answer": "",
        "contexts": [doc.page_content for doc in docs],
        "sources": [_source_label(doc) for doc in docs],
    }


def query(
    question: str,
    chat_history: list[dict] | None = None,
    company_filter: list[str] | None = None,
) -> dict:
    _load()
    docs = _hybrid_retrieve(question, company_filter=company_filter)
    history_text = _format_chat_history(chat_history)
    answer = _llm_chain.invoke({
        "context": _format_docs(docs),
        "question": question,
        "chat_history": history_text,
    })
    return {
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
        "sources": [_source_label(doc) for doc in docs],
    }


def stream_query(
    question: str,
    chat_history: list[dict] | None = None,
    company_filter: list[str] | None = None,
):
    _load()
    docs = _hybrid_retrieve(question, company_filter=company_filter)
    history_text = _format_chat_history(chat_history)
    for chunk in _llm_chain.stream({
        "context": _format_docs(docs),
        "question": question,
        "chat_history": history_text,
    }):
        yield chunk
