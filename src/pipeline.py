import json
import os
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
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")
TOP_K = 4
RRF_K = 60

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following excerpts from financial documents to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:""",
)

_faiss = None
_bm25 = None
_bm25_docs = None
_llm_chain = None


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
    tokenized = [doc.page_content.lower().split() for doc in _bm25_docs]
    _bm25 = BM25Okapi(tokenized)

    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0,
    )
    _llm_chain = _PROMPT | llm | StrOutputParser()


def _hybrid_retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """Reciprocal Rank Fusion over FAISS (semantic) + BM25 (keyword) results."""
    n_candidates = k * 10

    faiss_results = _faiss.similarity_search(question, k=n_candidates)

    tokens = question.lower().split()
    bm25_scores = _bm25.get_scores(tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    bm25_results = [_bm25_docs[i] for i in bm25_top_idx]

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(faiss_results):
        key = doc.page_content[:200]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:200]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        doc_map[key] = doc

    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[key] for key in ranked[:k]]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve(question: str) -> dict:
    _load()
    docs = _hybrid_retrieve(question)
    return {
        "answer": "",
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", doc.metadata.get("filename", "")) for doc in docs],
    }


def query(question: str) -> dict:
    _load()
    docs = _hybrid_retrieve(question)
    answer = _llm_chain.invoke({"context": _format_docs(docs), "question": question})
    return {
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", doc.metadata.get("filename", "")) for doc in docs],
    }


def stream_query(question: str):
    _load()
    docs = _hybrid_retrieve(question)
    for chunk in _llm_chain.stream({"context": _format_docs(docs), "question": question}):
        yield chunk
