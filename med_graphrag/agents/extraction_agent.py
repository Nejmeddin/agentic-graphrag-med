from __future__ import annotations

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from med_graphrag.llm.llm_client import get_llm
from med_graphrag.agents.extraction_schemas import ChunkExtractionResult


def build_extraction_chain() -> RunnableSerializable[Dict[str, Any], ChunkExtractionResult]:
    """
    Chaîne LangChain qui:
      - prend un dict {chunk_text, chunk_id, page, source}
      - appelle le LLM
      - renvoie un ChunkExtractionResult (Pydantic)
    """

    system_prompt = """You are a medical information extraction assistant.

You read short passages from the medical textbook
"Essentials of Human Diseases and Conditions".

Your task is to extract a list of medical entities mentioned in the text:
- DISEASE: diseases, disorders, conditions (e.g., hypertension, diabetes)
- SYMPTOM: patient symptoms or clinical signs (e.g., chest pain, fever)
- TREATMENT: medications, procedures, therapies (e.g., insulin, surgery)
- TEST: diagnostic tests, lab tests, imaging (e.g., ECG, MRI, blood test)

Guidelines:
- Only include entities that are clearly mentioned or strongly implied.
- Use simple canonical English names (e.g., 'heart attack' not 'myocardial infarction',
  unless the text itself uses the technical term).
- If you are unsure, set the type to OTHER or skip the entity.
- Keep short_definition very short (1-2 sentences maximum).
- Fill synonyms only when they are mentioned in the text or obviously standard.

Return your answer strictly following the provided JSON schema.
Do NOT include any text outside the JSON structure.
"""

    user_prompt = """Extract medical entities from the following chunk.

Chunk ID: {chunk_id}
Page: {page}
Source: {source}

Text:
\"\"\"{chunk_text}\"\"\""""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    llm = get_llm(temperature=0.0, max_tokens=512)

    # Structured output: LangChain converts LLM output -> ChunkExtractionResult
    structured_llm = llm.with_structured_output(ChunkExtractionResult)

    chain: RunnableSerializable[Dict[str, Any], ChunkExtractionResult] = prompt | structured_llm
    return chain
