import hashlib
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

OutletType = Literal["mainstream", "independent", "state_media", "unknown"]


class Article(BaseModel):
    id: str = Field(default="", description="SHA-256 of source_url — set automatically if empty.")
    title: str
    body: str
    source_name: str
    source_url: str
    published_at: Optional[datetime] = None
    outlet_type: OutletType = "unknown"
    query_used: str = Field(description="The search query that retrieved this article.")

    @model_validator(mode="after")
    def _set_id(self) -> "Article":
        if not self.id:
            self.id = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
        return self

    def to_retrieval_context(self) -> str:
        """Compact string representation for passing to downstream LLM workers."""
        date_str = self.published_at.date() if self.published_at else "unknown"
        return (
            f"[{self.source_name} | {date_str} | {self.outlet_type}]\n"
            f"Title: {self.title}\n"
            f"{self.body[:800].strip()}"
        )
