from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


CHROMA_DIR = Path("C:\\Users\\User\\Desktop\\agentic-graphrag-med\\data\\chroma")
CHROMA_COLLECTION_NAME = "medical_essentials"


def get_chroma_client(persist_directory: str | Path = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Create a persistent Chroma client. Data is stored on disk under data/chroma,
    so it survives across runs. :contentReference[oaicite:2]{index=2}
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(allow_reset=False),
    )
    return client


def get_or_create_collection(
    client: chromadb.Client,
    name: str = CHROMA_COLLECTION_NAME,
):
    """
    Get or create the Chroma collection for our medical chunks.

    By default, Chroma uses a SentenceTransformer embedding model (all-MiniLM-L6-v2),
    but here we explicitly set a SentenceTransformerEmbeddingFunction so it's clear. :contentReference[oaicite:3]{index=3}
    """
    # You can customize the model if you want (e.g. "all-mpnet-base-v2" or a medical model)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        collection = client.create_collection(
            name=name,
            embedding_function=embedding_function,
        )
    except chromadb.errors.UniqueConstraintError:
        # Already exists → reuse with same embedding function
        collection = client.get_collection(
            name=name,
            embedding_function=embedding_function,
        )

    return collection
