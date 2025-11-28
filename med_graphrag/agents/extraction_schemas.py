from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    TREATMENT = "TREATMENT"      # medication, procedure, therapy
    TEST = "TEST"                # lab test, imaging, diagnostic test
    OTHER = "OTHER"              # fallback


class MedicalEntity(BaseModel):
    name: str = Field(
        ...,
        description="Canonical name of the medical entity (e.g. 'hypertension').",
    )
    type: EntityType = Field(
        ...,
        description="Type of entity (DISEASE, SYMPTOM, TREATMENT, TEST, OTHER).",
    )
    short_definition: Optional[str] = Field(
        None,
        description="Very short definition or description.",
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations (if any).",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1.",
    )


class ChunkExtractionResult(BaseModel):
    """
    Résultat d'extraction pour UN chunk.
    """
    chunk_id: str = Field(..., description="Unique id of the chunk.")
    page: int = Field(..., description="Page number in the source PDF.")
    source: str = Field(..., description="Original file path or identifier.")
    entities: List[MedicalEntity] = Field(
        default_factory=list,
        description="List of extracted medical entities.",
    )
