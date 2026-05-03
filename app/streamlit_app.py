import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial Document Q&A", layout="wide")
st.title("Financial Document Q&A")
st.caption("RAG pipeline over FinQA financial reports · LangChain + FAISS + Gemma 4")

question = st.text_input(
    "Ask a question about financial documents",
    placeholder="What was the net revenue in the most recent quarter?",
)

if st.button("Ask", disabled=not question.strip()):
    try:
        # Retrieve source passages instantly (FAISS only, no LLM)
        sources_res = requests.post(
            f"{API_URL}/retrieve", json={"question": question}, timeout=30
        )
        sources_res.raise_for_status()
        sources_data = sources_res.json()

        # Stream the answer from the LLM
        st.subheader("Answer")
        answer_box = st.empty()
        full_answer = ""

        with requests.post(
            f"{API_URL}/query/stream",
            json={"question": question},
            stream=True,
            timeout=300,
        ) as stream_res:
            stream_res.raise_for_status()
            for chunk in stream_res.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    full_answer += chunk
                    answer_box.markdown(full_answer + "▌")

        answer_box.markdown(full_answer)

        # Show source passages
        st.subheader("Retrieved Source Passages")
        for i, (ctx, src) in enumerate(
            zip(sources_data["contexts"], sources_data["sources"]), 1
        ):
            label = f"Source {i}" + (f" — {src}" if src else "Unknown")
            with st.expander(label):
                st.write(ctx)

    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Run `uvicorn api.main:app --reload` first.")
    except Exception as e:
        st.error(f"Error: {e}")
