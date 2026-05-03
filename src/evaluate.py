import os
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.pipeline import query as rag_query

load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")

# Handcrafted questions over the indexed SEC 10-K filings
EVAL_QUESTIONS = [
    "What were JPMorgan's total net revenues in 2024?",
    "What are the main risk factors JPMorgan identifies in its annual report?",
    "How does Goldman Sachs describe its investment banking segment?",
    "What was Goldman Sachs's net earnings in its most recent fiscal year?",
    "How does MetLife describe its group benefits business?",
    "What are the primary sources of revenue for MetLife?",
    "What risk factors does Prudential Financial highlight for its insurance operations?",
    "How does AIG describe its general insurance business?",
    "What was AIG's adjusted pre-tax income in the most recent year?",
    "How do these companies describe their approach to managing interest rate risk?",
]


def run_evaluation() -> dict:
    print(f"Running RAGAS evaluation on {len(EVAL_QUESTIONS)} questions...\n")

    eval_samples = []
    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {q[:70]}...")
        result = rag_query(q)
        eval_samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
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

    # Faithfulness and AnswerRelevancy are reference-free — no ground truth needed
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    dataset = EvaluationDataset(samples=eval_samples)
    result = evaluate(dataset=dataset, metrics=metrics)
    print("\nResults:")
    print(result)
    return result


if __name__ == "__main__":
    run_evaluation()
