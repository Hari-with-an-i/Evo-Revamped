from pydantic import BaseModel, Field

class ContextSummary(BaseModel):
    who: str = Field(description="Key actors or persons involved in the claim.")
    what: str = Field(description="The core event or assertion.")
    when: str = Field(description="Inferred time window, ISO range or natural language.")
    competing_narratives: str = Field(
        default="",
        description=(
            "Pipe-separated alternative framings or counter-claims found in the articles. "
            "Example: 'WHO says no link|Telecom industry denies risk|Conspiracy blogs claim 5G activated virus'. "
            "Empty string if none."
        ),
    )
    claim_dimensions: str = Field(
        default="",
        description=(
            "Pipe-separated distinct verifiable sub-questions the claim raises (2–5 dimensions). "
            "Example: 'causal mechanism|timeline accuracy|source credibility|geographical scope'"
        ),
    )
