from typing import Literal
from pydantic import BaseModel, Field

TargetedTool = Literal["tavily", "gdelt_commoncrawl", "scholar_wiki"]


class TargetedQuery(BaseModel):
    query: str = Field(description="The search query string.")
    tool: TargetedTool = Field(description="Which retrieval worker should execute this query.")
    dimension: str = Field(default="", description="The claim dimension this query targets.")
