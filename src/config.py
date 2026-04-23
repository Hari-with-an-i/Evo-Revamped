import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM provider — Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Model selection — orchestrator/routing (use 8B for token efficiency)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    # Analyst narrative summary — kept on 70B for final report quality
    ANALYST_MODEL_NAME: str = os.getenv("ANALYST_MODEL_NAME", "llama-3.3-70b-versatile")
    # Classification tasks — evaluator + analyst frames/GDELT/shifts (own Groq rate-limit bucket)
    CLASSIFICATION_MODEL_NAME: str = os.getenv("CLASSIFICATION_MODEL_NAME", "llama-3.1-8b-instant")
    # Synthesis tasks — context_builder, researcher, writer (own Groq rate-limit bucket)
    SYNTHESIS_MODEL_NAME: str = os.getenv("SYNTHESIS_MODEL_NAME", "llama-3.1-8b-instant")

    # Agent behaviour
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "50"))

    # Retrieval thresholds
    MIN_ARTICLES: int = int(os.getenv("MIN_ARTICLES", "10"))
    MAX_RETRIEVAL_LOOPS: int = int(os.getenv("MAX_RETRIEVAL_LOOPS", "3"))

    # Evaluation thresholds (configurable for token scaling)
    MAX_ARTICLES_FOR_NLI: int = int(os.getenv("MAX_ARTICLES_FOR_NLI", "15"))
    SENTIMENT_DELTA_THRESHOLD: float = float(os.getenv("SENTIMENT_DELTA_THRESHOLD", "0.3"))
    # SBERT relevance filter thresholds
    RELEVANCE_DROP: float = float(os.getenv("RELEVANCE_DROP", "0.35"))
    RELEVANCE_PERIPHERAL: float = float(os.getenv("RELEVANCE_PERIPHERAL", "0.5"))
    # Output format
    REPORT_INCLUDE_JSON: bool = os.getenv("REPORT_INCLUDE_JSON", "false").lower() == "true"

    # Logging — see src/logger.py
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")   # "json" | "text"
    LOG_FILE: str = os.getenv("LOG_FILE", "")            # empty = stdout only


config = Config()
