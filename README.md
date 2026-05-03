# Financial Document Q&A

A RAG pipeline that lets you ask natural language questions over real SEC 10-K filings. Built it to get hands-on with retrieval-augmented generation using actual financial data rather than toy datasets.

Currently indexes annual reports from JPMorgan, Goldman Sachs, MetLife, Prudential, and AIG. Runs fully local — no OpenAI, no cloud LLM calls.

---

## How it works

Questions go through three steps:

1. The query gets embedded using `all-MiniLM-L6-v2` and matched against a FAISS index of chunked 10-K text
2. The top 4 passages get stuffed into a prompt
3. Gemma 4 (running locally via LM Studio) generates the answer

The answer streams back token by token so you're not just staring at a spinner waiting for a 26B model.

Evaluation is done with RAGAS — faithfulness, answer relevancy, and context recall — using the ground truth Q&A pairs from the dataset.

---

## Stack

- **LangChain** (LCEL) for the retrieval chain
- **FAISS** for vector search
- **sentence-transformers** for embeddings
- **FastAPI** for the backend
- **Streamlit** for the frontend
- **RAGAS** for evaluation
- **LM Studio** for running Gemma 4 locally

---

## Setup

**Requirements:** Python 3.10+, LM Studio running locally with `google/gemma-4-26b-a4b` loaded

```bash
pip install -r requirements.txt
```

Download SEC filings:
```bash
python src/download_sec.py
```

Build the FAISS index:
```bash
python src/ingest.py
```

Start the API and frontend in separate terminals:
```bash
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

Copy `.env.example` to `.env` and update `LM_STUDIO_BASE_URL` if your LM Studio isn't on the default port.

---

## Evaluation

Runs RAGAS metrics on 10 samples from the dev set:

```bash
python src/evaluate.py
```

Outputs faithfulness, answer relevancy, and context recall scores. Uses the same local LM Studio model so no external API calls.

---

## Project structure

```
api/          FastAPI backend
app/          Streamlit frontend
src/
  download_sec.py   pulls 10-K filings from SEC EDGAR
  ingest.py         chunks + embeds + builds FAISS index
  pipeline.py       RAG chain
  evaluate.py       RAGAS evaluation
data/         SEC filings (not committed)
vectorstore/  FAISS index (not committed)
```

---

## What's next

- Hybrid search (BM25 + FAISS) to improve retrieval on exact company/ticker names
- RAGAS metrics panel in the UI
- Conversational memory for follow-up questions
- Company filter in the sidebar
- Azure deployment config
