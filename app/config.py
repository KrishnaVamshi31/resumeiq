"""Typed application configuration, loaded once from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars or a local `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # --- Service ---------------------------------------------------------
    env: Literal["development", "staging", "production"] = Field(
        default="development", alias="RESUMEIQ_ENV"
    )
    log_level: str = Field(default="INFO", alias="RESUMEIQ_LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(default="json", alias="RESUMEIQ_LOG_FORMAT")
    host: str = Field(default="0.0.0.0", alias="RESUMEIQ_HOST")
    # Every PaaS (Render, Railway, Fly, Heroku) injects the port to bind as
    # `PORT` and routes traffic only to that port. It is checked first so a
    # deployment works with no platform-specific configuration, while
    # `RESUMEIQ_PORT` stays available for local use.
    port: int = Field(
        default=8000, validation_alias=AliasChoices("PORT", "RESUMEIQ_PORT")
    )
    cors_origins: str = Field(default="http://localhost:8501", alias="RESUMEIQ_CORS_ORIGINS")

    # --- Storage ---------------------------------------------------------
    database_url: str = Field(default="sqlite:///./data/resumeiq.db", alias="RESUMEIQ_DATABASE_URL")

    # --- Uploads ---------------------------------------------------------
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, alias="RESUMEIQ_MAX_UPLOAD_BYTES")
    max_resume_chars: int = Field(default=60_000, alias="RESUMEIQ_MAX_RESUME_CHARS")

    # --- LLM -------------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_enabled: bool = Field(default=True, alias="RESUMEIQ_LLM_ENABLED")
    llm_model: str = Field(default="claude-opus-5", alias="RESUMEIQ_LLM_MODEL")
    llm_max_tokens: int = Field(default=8000, alias="RESUMEIQ_LLM_MAX_TOKENS")
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="medium", alias="RESUMEIQ_LLM_EFFORT"
    )
    llm_timeout_seconds: float = Field(default=60.0, alias="RESUMEIQ_LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="RESUMEIQ_LLM_MAX_RETRIES")
    llm_redact_pii: bool = Field(default=True, alias="RESUMEIQ_LLM_REDACT_PII")

    # --- Rate limiting ---------------------------------------------------
    rate_limit_requests: int = Field(default=60, alias="RESUMEIQ_RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RESUMEIQ_RATE_LIMIT_WINDOW_SECONDS")

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_available(self) -> bool:
        """True when the narrative coaching layer can actually reach the API."""
        return self.llm_enabled and bool(self.anthropic_api_key)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so config is parsed exactly once per process."""
    return Settings()
