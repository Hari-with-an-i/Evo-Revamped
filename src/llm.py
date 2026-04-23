from langchain_groq import ChatGroq

from src.config import config


def get_classification_llm():
    """meta-llama/llama-4-scout-17b-16e-instruct claims/labels/source-type, analyst frames/GDELT/shifts."""
    return ChatGroq(model=config.CLASSIFICATION_MODEL_NAME, api_key=config.GROQ_API_KEY)


def get_synthesis_llm():
    """llama-3.1-8b-instant — context_builder, researcher, writer."""
    return ChatGroq(model=config.SYNTHESIS_MODEL_NAME, api_key=config.GROQ_API_KEY)


# Backwards-compat alias — remove once all callers are updated
def get_small_llm():
    return get_synthesis_llm()
