import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

SEC_DIR = Path("data/sec_filings/sec-edgar-filings")
VECTORSTORE_DIR = Path("vectorstore")

# Cap text per filing to keep ingest fast (~1M chars ≈ 200k tokens)
MAX_CHARS_PER_FILING = 1_000_000


def _extract_primary_doc(filepath: Path) -> str:
    """Pull the primary 10-K HTML block out of an EDGAR full-submission.txt."""
    in_doc = False
    is_primary = False
    in_text = False
    lines = []

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "<DOCUMENT>":
                in_doc = True
                is_primary = False
                in_text = False
                lines = []
            elif stripped == "</DOCUMENT>":
                if is_primary and lines:
                    return "\n".join(lines)
                in_doc = False
            elif in_doc:
                if stripped.startswith("<TYPE>"):
                    is_primary = stripped[6:].strip() == "10-K"
                elif stripped == "<TEXT>":
                    in_text = True
                elif stripped == "</TEXT>":
                    in_text = False
                elif in_text and is_primary:
                    lines.append(line.rstrip())
    return ""


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_sec_filings() -> list[Document]:
    docs = []
    for submission in SEC_DIR.rglob("full-submission.txt"):
        parts = submission.parts
        ticker = parts[parts.index("sec-edgar-filings") + 1] if "sec-edgar-filings" in parts else "unknown"
        accession = submission.parent.name

        print(f"  Parsing {ticker} / {accession}...")
        raw = _extract_primary_doc(submission)
        if not raw:
            print(f"    Skipped — no primary 10-K block found")
            continue

        text = _html_to_text(raw)[:MAX_CHARS_PER_FILING]
        if len(text) < 1000:
            print(f"    Skipped — extracted text too short")
            continue

        docs.append(Document(
            page_content=text,
            metadata={"ticker": ticker, "accession": accession, "source": f"{ticker}/10-K"},
        ))
        print(f"    OK — {len(text):,} chars")

    return docs


def build_index():
    print("Loading SEC 10-K filings...")
    all_docs = load_sec_filings()

    if not all_docs:
        raise FileNotFoundError(
            f"No filings found in {SEC_DIR}. Run `python src/download_sec.py` first."
        )

    print(f"\nLoaded {len(all_docs)} filings")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"FAISS index saved to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    build_index()
