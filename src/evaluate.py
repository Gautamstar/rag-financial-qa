import json
import os
from pathlib import Path
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.pipeline import query as rag_query

load_dotenv()

N_SAMPLES = 10
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")


def run_evaluation() -> dict:
    dev_path = Path("data/dev.json")
    if not dev_path.exists():
        raise FileNotFoundError("data/dev.json not found — drop it in the data/ folder.")

    with open(dev_path) as f:
        dev_data = json.load(f)

    samples = [e for e in dev_data if e.get("qa", {}).get("answer")][:N_SAMPLES]

    eval_samples = []
    for i, entry in enumerate(samples):
        q = entry["qa"]["question"]
        gt = str(entry["qa"]["answer"])
        print(f"[{i+1}/{len(samples)}] {q[:70]}...")
        result = rag_query(q)
        eval_samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=gt,
        ))

    llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
    ]

    dataset = EvaluationDataset(samples=eval_samples)
    result = evaluate(dataset=dataset, metrics=metrics)
    print(result)
    return result


if __name__ == "__main__":
    run_evaluation()
