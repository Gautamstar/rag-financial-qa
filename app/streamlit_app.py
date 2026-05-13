import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

COMPANIES = {
    "JPMorgan Chase": "JPM",
    "Goldman Sachs": "GS",
    "MetLife": "MET",
    "Prudential Financial": "PRU",
    "AIG": "AIG",
}

st.set_page_config(page_title="Financial Document Q&A", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")
    selected_companies = st.multiselect(
        "Company",
        options=list(COMPANIES.keys()),
        default=list(COMPANIES.keys()),
        help="Restrict retrieval to selected filings. Deselect all to search across everything.",
    )
    company_filter = [COMPANIES[c] for c in selected_companies] if selected_companies else None

    st.divider()
    st.header("Generation")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Higher = more creative answers. Lower = more deterministic.",
    )
    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Number of document passages retrieved per query.",
    )

    st.divider()
    st.header("Evaluation")
    st.caption("Runs RAGAS faithfulness + answer relevancy over 10 built-in questions. Takes several minutes.")
    if st.button("Run Evaluation", use_container_width=True):
        with st.spinner("Running RAGAS evaluation…"):
            try:
                res = requests.post(f"{API_URL}/metrics", timeout=900)
                res.raise_for_status()
                metrics = res.json()
                st.metric("Faithfulness", f"{metrics.get('faithfulness', 'N/A'):.3f}" if metrics.get("faithfulness") is not None else "N/A")
                st.metric("Answer Relevancy", f"{metrics.get('answer_relevancy', 'N/A'):.3f}" if metrics.get("answer_relevancy") is not None else "N/A")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API.")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat interface ────────────────────────────────────────────────────────────

st.title("Financial Document Q&A")
st.caption("RAG pipeline over SEC 10-K filings · LangChain + FAISS + BM25 + Gemma 4")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for i, (ctx, src) in enumerate(zip(msg["contexts"], msg["sources"]), 1):
                    label = f"Source {i}" + (f" — {src}" if src else "")
                    st.markdown(f"**{label}**")
                    st.caption(ctx[:600] + ("…" if len(ctx) > 600 else ""))

question = st.chat_input("Ask a question about the financial filings…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # History sent to the API excludes the current question (already in `question`)
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        answer_box = st.empty()
        full_answer = ""

        try:
            # Retrieve source passages first (no LLM call)
            sources_res = requests.post(
                f"{API_URL}/retrieve",
                json={"question": question, "company_filter": company_filter, "top_k": top_k},
                timeout=30,
            )
            sources_res.raise_for_status()
            sources_data = sources_res.json()

            # Stream the answer
            with requests.post(
                f"{API_URL}/query/stream",
                json={
                    "question": question,
                    "chat_history": api_history or None,
                    "company_filter": company_filter,
                    "temperature": temperature,
                    "top_k": top_k,
                },
                stream=True,
                timeout=300,
            ) as stream_res:
                stream_res.raise_for_status()
                for chunk in stream_res.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_answer += chunk
                        answer_box.markdown(full_answer + "▌")

            answer_box.markdown(full_answer)

            with st.expander("Retrieved sources"):
                for i, (ctx, src) in enumerate(
                    zip(sources_data["contexts"], sources_data["sources"]), 1
                ):
                    label = f"Source {i}" + (f" — {src}" if src else "")
                    st.markdown(f"**{label}**")
                    st.caption(ctx[:600] + ("…" if len(ctx) > 600 else ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources_data["sources"],
                "contexts": sources_data["contexts"],
            })

        except requests.exceptions.ConnectionError:
            answer_box.error("Cannot reach the API. Run `uvicorn api.main:app --reload` first.")
        except Exception as e:
            answer_box.error(f"Error: {e}")
