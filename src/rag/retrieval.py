from pathlib import Path
import os

from dotenv import load_dotenv
import chromadb

from src.rag.embeddings import get_embedding

import textwrap

load_dotenv()

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chroma"))
COLLECTION_NAME = "company_reports"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB collection not found. Run: uv run python -m src.rag.ingestion"
        ) from exc


def search_knowledge_base(
    query: str,
    ticker: str,
    n_results: int = 5,
) -> str:
    collection = get_collection()

    ticker = ticker.upper()
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"ticker": ticker},
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return f"No RAG results found for ticker: {ticker}"

    output = [
        "## RAG Results",
        "",
        f"Query: {query}",
        f"Ticker: {ticker}",
        "",
    ]

    for index, document in enumerate(documents):
        metadata = metadatas[index]
        distance = distances[index] if index < len(distances) else None

        output.append(f"### Result {index + 1}")
        output.append(f"Source: {metadata.get('source_file')}")
        output.append(f"Company: {metadata.get('company_name')}")
        output.append(f"Fiscal period: {metadata.get('fiscal_period')}")
        output.append(f"Chunk: {metadata.get('chunk_index')}")

        if distance is not None:
            output.append(f"Distance: {distance:.4f}")

        output.append("")
        output.append(textwrap.fill(document, width=100))
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    print(
        search_knowledge_base(
            query="semiconductor demand and export restrictions",
            ticker="ASML",
            n_results=3,
        )
    )