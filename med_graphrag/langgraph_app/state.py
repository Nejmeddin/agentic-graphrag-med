from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from med_graphrag.planning.planner_schemas import RetrievalPlan
from med_graphrag.retrieval.combined_retriever import RetrievedContext


class QAState(BaseModel):
    """
    State global échangé entre les nodes LangGraph.
    """
    question: str
    plan: Optional[RetrievalPlan] = None
    retrieved_context: Optional[RetrievedContext] = None
    answer: Optional[str] = None
