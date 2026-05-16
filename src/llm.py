from langchain_groq import ChatGroq

from src.config import config


def get_classification_llm():
    """claims/labels/source-type, analyst frames/GDELT/shifts — short output tasks."""
    return ChatGroq(model=config.CLASSIFICATION_MODEL_NAME, api_key=config.GROQ_API_KEY, max_tokens=1000, temperature=0)


def get_synthesis_llm():
    """context_builder, researcher, writer — prose output tasks."""
    return ChatGroq(model=config.SYNTHESIS_MODEL_NAME, api_key=config.GROQ_API_KEY, max_tokens=1200)


# Backwards-compat alias — remove once all callers are updated
def get_small_llm():
    return get_synthesis_llm()
