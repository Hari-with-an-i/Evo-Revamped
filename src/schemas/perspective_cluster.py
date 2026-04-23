from typing import Literal
from pydantic import BaseModel, Field

CorroborationLevel = Literal["unverified", "partial", "corroborated"]


class PerspectiveCluster(BaseModel):
    label: str = Field(description="LLM-generated plain-language viewpoint, 1–2 sentences.")
    key_claims: list[str] = Field(description="Key factual claims extracted from articles in this cluster.")
    article_ids: list[str] = Field(description="Article IDs belonging to this cluster.")
    domain_roots: list[str] = Field(description="Unique domain roots of articles in this cluster.")
    corroboration_count: int = Field(description="Number of independent domain roots.")
    corroboration_level: CorroborationLevel = Field(
        description="unverified=1–2 roots, partial=3–4, corroborated=5+."
    )
    source_scope: Literal["secondary"] = Field(
        default="secondary",
        description="Always 'secondary' — clusters are built from secondary sources only.",
    )
