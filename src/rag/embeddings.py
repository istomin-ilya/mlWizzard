import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_embedding(text: str) -> list[float]:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env before running RAG."
        )

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        model=model,
        input=text,
    )

    return response.data[0].embedding