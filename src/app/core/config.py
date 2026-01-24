"""Configuration management for the multi-agent RAG system.

This module uses Pydantic Settings to load and validate environment variables
for OpenAI models, Pinecone settings, and other system parameters.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model_name: str = "gpt-4o-mini"

    # ✅ FIX: must match Pinecone index dimension (1536)
    openai_embedding_model_name: str = "text-embedding-3-small"

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: str

    # Retrieval Configuration
    retrieval_k: int = 4

    # No .env file (use terminal env vars)
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
