from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

CHROMA_DIR = Path("data/chroma")
CHROMA_COLLECTION_NAME = "medical_essentials"


def get_chroma_client(persist_directory: str | Path = CHROMA_DIR) -> chromadb.PersistentClient:
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

    Uses get_or_create_collection so it's safe even if the collection
    already exists.
    """
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
    )
    return collection
