"""Application settings loaded from environment / `.env`."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration. Reads from `.env` and OS environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    chroma_path: Path = Path("./data/chroma")
    pdf_dir: Path = Path("./data/pdfs")
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4
    log_level: str = "INFO"
    collection_name: str = "papers"


settings = Settings()
