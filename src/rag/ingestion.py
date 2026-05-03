from pathlib import Path
import os

from dotenv import load_dotenv
import chromadb

from src.rag.chunking import chunk_text, load_reports
from src.rag.embeddings import get_embedding

load_dotenv()

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "data/chroma"))
COLLECTION_NAME = "company_reports"


def build_chroma_index() -> None:
    reports = load_reports("data/reports")

    if not reports:
        raise RuntimeError("No reports found in data/reports")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for report in reports:
        chunks = chunk_text(report["text"])
        base_metadata = report["metadata"]

        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(base_metadata)
            metadata["chunk_index"] = chunk_index

            doc_id = (
                f"{metadata['ticker']}_"
                f"{metadata['fiscal_period']}_"
                f"{chunk_index}"
            )

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(metadata)
            embeddings.append(get_embedding(chunk))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("RAG index built successfully")
    print(f"Path: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Reports loaded: {len(reports)}")
    print(f"Chunks indexed: {len(documents)}")


if __name__ == "__main__":
    build_chroma_index()