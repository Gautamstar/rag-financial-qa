import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

VECTORSTORE_DIR = Path("vectorstore")
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following excerpts from financial documents to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:""",
)

_retriever = None
_chain = None


def _load():
    global _retriever, _chain
    if _chain is not None:
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0,
    )

    _chain = (
        {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query(question: str) -> dict:
    _load()
    docs = _retriever.invoke(question)
    answer = _chain.invoke(question)
    return {
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", doc.metadata.get("filename", "")) for doc in docs],
    }


def retrieve(question: str) -> dict:
    _load()
    docs = _retriever.invoke(question)
    return {
        "answer": "",
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", doc.metadata.get("filename", "")) for doc in docs],
    }


def stream_query(question: str):
    _load()
    for chunk in _chain.stream(question):
        yield chunk
