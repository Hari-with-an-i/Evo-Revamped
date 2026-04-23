from typing import Literal
from pydantic import BaseModel, Field


class ParsedInput(BaseModel):
    claim: str = Field(description="The core falsifiable claim, normalized to a single declarative sentence.")
    entities: list[str] = Field(default_factory=list, description="Key named entities: people, orgs, locations.")
    timeframe: str | None = Field(default=None, description="Date or date range hint extracted from the input, ISO format where possible.")
    input_type: Literal["claim", "raw_text"]
