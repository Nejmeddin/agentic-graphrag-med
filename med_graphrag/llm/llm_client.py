from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # pip install langchain-groq

load_dotenv(dotenv_path="C:\\Users\\User\\Desktop\\agentic-graphrag-med\\data\\source\\.env")
def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> ChatGroq:
    """
    Retourne une instance de ChatGroq configurée depuis les variables d'environnement.
    On copie un peu l'idée de ton notebook: ChatGroq + llama-3.3-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment / .env")

    if model_name is None:
        model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm