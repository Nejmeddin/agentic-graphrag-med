from __future__ import annotations

import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from med_graphrag.llm.llm_client import get_llm
from med_graphrag.planning.planner_schemas import RetrievalPlan, RetrievalMode
from pydantic import ValidationError


def build_planner_chain():
    """
    LLM qui lit la question utilisateur et renvoie un JSON décrivant
    la stratégie de retrieval (mode, n_results, neighbor_hops, use_graph).
    On reste en 'raw JSON' pour éviter les problèmes de tools avec Groq.
    """

    system_prompt = """You are a planning agent for a medical GraphRAG system.

You MUST select the best retrieval strategy for each user question.
The knowledge comes from a medical textbook
("Essentials of Human Diseases and Conditions").

Available retrieval modes:

- SIMPLE_DEFINITION:
  The question asks for a definition or basic explanation of a single disease,
  condition, or symptom.
  Example: "What is cystic fibrosis?", "Explain hypertension".

- GRAPH_RELATION:
  The question asks about relationships between diseases, complications,
  risk factors, systems, or progression across multiple conditions.
  Example: "Which diseases can complicate pregnancy?", "How is diabetes related to kidney disease?".

- COMPARE_DISEASES:
  The question explicitly compares diseases or asks for differences/similarities.
  Example: "Compare type 1 and type 2 diabetes".

- UNSURE:
  The question is unclear, non-medical, or cannot be answered from this textbook.

You MUST output a single JSON object with the following schema:

{{
  "mode": "SIMPLE_DEFINITION" | "GRAPH_RELATION" | "COMPARE_DISEASES" | "UNSURE",
  "n_results": <int between 1 and 15>,
  "neighbor_hops": <int between 0 and 3>,
  "use_graph": <true or false>,
  "reason": "<short explanation>"
}}

Guidelines:
- SIMPLE_DEFINITION → usually n_results ≈ 3–5, neighbor_hops = 0 or 1, use_graph can be false.
- GRAPH_RELATION   → usually n_results ≈ 5–8, neighbor_hops = 1–2, use_graph = true.
- COMPARE_DISEASES → usually n_results ≈ 5–8, neighbor_hops = 1–2, use_graph = true.
- If UNSURE, set n_results=3, neighbor_hops=0, use_graph=false.

Do NOT include any extra keys.
Do NOT add comments or natural language before or after the JSON.
"""


    user_prompt = """User medical question:

{question}

Decide the best retrieval plan and output ONLY the JSON object."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    llm = get_llm(temperature=0.0, max_tokens=256)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain


def plan_retrieval(question: str) -> RetrievalPlan:
    """
    Appelle le planner LLM et parse le JSON en RetrievalPlan (Pydantic).
    Si le parsing échoue, on retourne un plan par défaut.
    """
    chain = build_planner_chain()
    raw = chain.invoke({"question": question})
    raw_str = raw.strip()

    try:
        data = json.loads(raw_str)
        plan = RetrievalPlan.model_validate(data)
        return plan
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"⚠️ Planner JSON/validation error: {e}")
        print(f"   Raw planner output (truncated): {raw_str[:200]!r}")
        # Plan de secours simple
        return RetrievalPlan(
            mode=RetrievalMode.SIMPLE_DEFINITION,
            n_results=5,
            neighbor_hops=1,
            use_graph=True,
            reason="Fallback plan due to parsing error.",
        )
