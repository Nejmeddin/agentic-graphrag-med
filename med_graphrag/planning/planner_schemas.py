from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    SIMPLE_DEFINITION = "SIMPLE_DEFINITION"      # définition d’une seule maladie / condition
    GRAPH_RELATION = "GRAPH_RELATION"            # relations entre maladies, symptômes, complications
    COMPARE_DISEASES = "COMPARE_DISEASES"        # comparer plusieurs maladies
    UNSURE = "UNSURE"                            # question ambiguë ou hors scope


class RetrievalPlan(BaseModel):
    mode: RetrievalMode = Field(
        ...,
        description="Chosen strategy for this question.",
    )
    n_results: int = Field(
        5,
        ge=1,
        le=15,
        description="Number of top chunks to retrieve from vector search.",
    )
    neighbor_hops: int = Field(
        1,
        ge=0,
        le=3,
        description="How many NEXT_CHUNK hops to use in the graph.",
    )
    use_graph: bool = Field(
        True,
        description="Whether to expand context using the graph (neighbors + entities).",
    )
    reason: Optional[str] = Field(
        None,
        description="Short natural language justification of the chosen plan.",
    )
