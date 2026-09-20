"""newsscout.config
~~~~~~~~~~~~~~~~~~
Centralized application configuration using Pydantic Settings v2.
Reads from environment variables and optional .env file with type safety,
secret masking, and platform defaults for Raspberry Pi 5.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """NewsScout configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ==========================================
    # Application & Environment
    # ==========================================
    app_name: str = "NewsScout"
    environment: Literal["development", "production", "test"] = Field(
        default="development",
        validation_alias=AliasChoices("NEWSSCOUT_ENVIRONMENT", "ENVIRONMENT", "APP_ENV", "ENV"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("NEWSSCOUT_LOG_LEVEL", "LOG_LEVEL"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("NEWSSCOUT_HOST", "HOST"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("NEWSSCOUT_PORT", "PORT"),
    )
    timezone: str = Field(
        default="Europe/Berlin",
        validation_alias=AliasChoices("NEWSSCOUT_TIMEZONE", "NEWSSCOUT_TZ", "TZ", "BRIEFING_TIMEZONE", "TIMEZONE"),
    )

    # ==========================================
    # SQLite Database Storage
    # ==========================================
    db_path: Path = Field(
        default=Path("data/ai_scout.db"),
        validation_alias=AliasChoices("NEWSSCOUT_DB_PATH", "DB_PATH", "SQLITE_DB_PATH"),
    )
    sqlite_busy_timeout_ms: int = Field(
        default=5000,
        validation_alias=AliasChoices("NEWSSCOUT_SQLITE_BUSY_TIMEOUT", "SQLITE_BUSY_TIMEOUT", "SQLITE_BUSY_TIMEOUT_MS"),
    )
    sqlite_cache_size_kb: int = Field(
        default=-64000,  # 64MB cache in SQLite negative KiB notation
        validation_alias=AliasChoices("NEWSSCOUT_SQLITE_CACHE_SIZE", "SQLITE_CACHE_SIZE"),
    )

    # ==========================================
    # API Credentials (Masked Secrets)
    # ==========================================
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_GEMINI_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        validation_alias=AliasChoices("NEWSSCOUT_GEMINI_MODEL", "GEMINI_MODEL"),
    )

    # ==========================================
    # Universal LLM Provider Configuration (R3)
    # ==========================================
    llm_provider: Literal["gemini", "openai_compatible", "mock"] = Field(
        default="gemini",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_PROVIDER", "LLM_PROVIDER"),
    )
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_BASE_URL", "LLM_BASE_URL"),
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_MODEL", "LLM_MODEL"),
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_LLM_API_KEY", "LLM_API_KEY"),
    )
    llm_custom_headers: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("NEWSSCOUT_LLM_CUSTOM_HEADERS", "LLM_CUSTOM_HEADERS"),
    )
    llm_temperature: float = Field(
        default=0.1,
        validation_alias=AliasChoices("NEWSSCOUT_LLM_TEMPERATURE", "LLM_TEMPERATURE"),
    )
    llm_timeout_seconds: float = Field(
        default=45.0,
        validation_alias=AliasChoices("NEWSSCOUT_LLM_TIMEOUT", "LLM_TIMEOUT_SECONDS"),
    )
    llm_fallback_provider: Literal["gemini", "openai_compatible", "none", "mock"] = Field(
        default="openai_compatible",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_PROVIDER"),
    )
    llm_fallback_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_BASE_URL"),
    )
    llm_fallback_model: str = Field(
        default="glm-4-flash",
        validation_alias=AliasChoices("NEWSSCOUT_LLM_FALLBACK_MODEL", "LLM_FALLBACK_MODEL"),
    )
    llm_fallback_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_LLM_FALLBACK_API_KEY", "LLM_FALLBACK_API_KEY"),
    )

    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"),
    )
    telegram_chat_id: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSSCOUT_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID"),
    )
    github_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_GITHUB_TOKEN", "GITHUB_TOKEN"),
    )

    # ==========================================
    # Multi-Messenger Gateways (R1)
    # ==========================================
    delivery_channels: list[str] = Field(
        default=["telegram"],
        validation_alias=AliasChoices("NEWSSCOUT_DELIVERY_CHANNELS", "DELIVERY_CHANNELS"),
    )
    delivery_timeout_seconds: float = Field(
        default=15.0,
        validation_alias=AliasChoices("NEWSSCOUT_DELIVERY_TIMEOUT_SECONDS", "DELIVERY_TIMEOUT_SECONDS", "DELIVERY_TIMEOUT"),
    )
    # WhatsApp (WAHA / Baileys bridge)
    whatsapp_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_ENABLED", "WHATSAPP_ENABLED"),
    )
    whatsapp_bridge_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_BRIDGE_URL", "WHATSAPP_BRIDGE_URL"),
    )
    whatsapp_bridge_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_BRIDGE_TOKEN", "WHATSAPP_BRIDGE_TOKEN"),
    )
    whatsapp_session: str = Field(
        default="default",
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_SESSION", "WHATSAPP_SESSION"),
    )
    whatsapp_recipient_id: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_RECIPIENT_ID", "WHATSAPP_RECIPIENT_ID", "WHATSAPP_RECIPIENT"),
    )
    whatsapp_recipients: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("NEWSSCOUT_WHATSAPP_RECIPIENTS", "WHATSAPP_RECIPIENTS"),
    )
    # Signal (signal-cli-rest-api sidecar)
    signal_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("NEWSSCOUT_SIGNAL_ENABLED", "SIGNAL_ENABLED"),
    )
    signal_bridge_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices("NEWSSCOUT_SIGNAL_BRIDGE_URL", "SIGNAL_BRIDGE_URL", "NEWSSCOUT_SIGNAL_API_URL", "SIGNAL_API_URL"),
    )
    signal_sender_number: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSSCOUT_SIGNAL_SENDER_NUMBER", "SIGNAL_SENDER_NUMBER", "NEWSSCOUT_SIGNAL_NUMBER", "SIGNAL_NUMBER"),
    )
    signal_recipient_id: str = Field(
        default="",
        validation_alias=AliasChoices("NEWSSCOUT_SIGNAL_RECIPIENT_ID", "SIGNAL_RECIPIENT_ID", "SIGNAL_RECIPIENT"),
    )
    signal_recipients: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("NEWSSCOUT_SIGNAL_RECIPIENTS", "SIGNAL_RECIPIENTS"),
    )

    # ==========================================
    # Web Search Pipeline Configuration (R2)
    # ==========================================
    searxng_base_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices("NEWSSCOUT_SEARXNG_BASE_URL", "SEARXNG_BASE_URL"),
    )
    searxng_categories: str = Field(
        default="general,it",
        validation_alias=AliasChoices("NEWSSCOUT_SEARXNG_CATEGORIES", "SEARXNG_CATEGORIES"),
    )
    searxng_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWSSCOUT_SEARXNG_ENABLED", "SEARXNG_ENABLED"),
    )
    duckduckgo_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWSSCOUT_DUCKDUCKGO_ENABLED", "DUCKDUCKGO_ENABLED"),
    )
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_TAVILY_API_KEY", "TAVILY_API_KEY"),
    )
    tavily_search_depth: str = Field(
        default="basic",
        validation_alias=AliasChoices("NEWSSCOUT_TAVILY_SEARCH_DEPTH", "TAVILY_SEARCH_DEPTH"),
    )
    exa_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEWSSCOUT_EXA_API_KEY", "EXA_API_KEY"),
    )
    exa_search_type: str = Field(
        default="neural",
        validation_alias=AliasChoices("NEWSSCOUT_EXA_SEARCH_TYPE", "EXA_SEARCH_TYPE"),
    )
    search_timeout_seconds: float = Field(
        default=8.0,
        validation_alias=AliasChoices("NEWSSCOUT_SEARCH_TIMEOUT_SECONDS", "SEARCH_TIMEOUT_SECONDS", "SEARCH_TIMEOUT"),
    )
    search_max_results_per_engine: int = Field(
        default=10,
        validation_alias=AliasChoices("NEWSSCOUT_SEARCH_MAX_RESULTS_PER_ENGINE", "SEARCH_MAX_RESULTS_PER_ENGINE"),
    )
    search_max_queries_per_cycle: int = Field(
        default=5,
        validation_alias=AliasChoices("NEWSSCOUT_SEARCH_MAX_QUERIES_PER_CYCLE", "SEARCH_MAX_QUERIES_PER_CYCLE"),
    )

    # ==========================================
    # Scheduling & Timing (24-Hour Format)
    # ==========================================
    schedule_morning: str = Field(
        default="07:00",
        validation_alias=AliasChoices("NEWSSCOUT_SCHEDULE_MORNING", "SCHEDULE_MORNING"),
    )
    schedule_afternoon: str = Field(
        default="16:00",
        validation_alias=AliasChoices("NEWSSCOUT_SCHEDULE_AFTERNOON", "SCHEDULE_AFTERNOON"),
    )
    pipeline_interval_hours: int = Field(
        default=2,
        validation_alias=AliasChoices("NEWSSCOUT_PIPELINE_INTERVAL_HOURS", "PIPELINE_INTERVAL_HOURS"),
    )

    # ==========================================
    # Audio Synthesis & Voices (edge-tts)
    # ==========================================
    audio_output_dir: Path = Field(
        default=Path("data/audio"),
        validation_alias=AliasChoices("NEWSSCOUT_AUDIO_OUTPUT_DIR", "AUDIO_OUTPUT_DIR"),
    )
    tts_voice_male: str = Field(
        default="de-DE-ConradNeural",
        validation_alias=AliasChoices("NEWSSCOUT_TTS_VOICE_MALE", "TTS_VOICE_MALE"),
    )
    tts_voice_female: str = Field(
        default="de-DE-KatjaNeural",
        validation_alias=AliasChoices("NEWSSCOUT_TTS_VOICE_FEMALE", "TTS_VOICE_FEMALE"),
    )
    audio_min_duration_minutes: int = 15
    audio_max_duration_minutes: int = 30
    audio_retention_days: int = 14

    # ==========================================
    # Preferences & Few-Shot Learning
    # ==========================================
    preferences_file: Path = Field(
        default=Path("data/preferences.json"),
        validation_alias=AliasChoices("NEWSSCOUT_PREFERENCES_FILE", "PREFERENCES_FILE"),
    )
    few_shot_exemplars_count: int = 5

    # ==========================================
    # Ingestion & Filtering Thresholds
    # ==========================================
    hn_min_score: int = 120
    hn_min_comment_ratio: float = 0.4
    hf_papers_min_upvotes: int = 5
    core_profile_ratio: float = 0.8
    serendipity_profile_ratio: float = 0.2
    min_breakthrough_score: float = 7.0
    min_roi_score: float = 7.0

    def ensure_directories(self) -> None:
        """Ensures all configured filesystem directories exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Returns True if running in production mode."""
        return self.environment == "production"

    @property
    def has_gemini_credentials(self) -> bool:
        """Checks whether a valid Gemini API key is configured."""
        return bool(self.gemini_api_key.get_secret_value().strip())

    @property
    def has_llm_credentials(self) -> bool:
        """Checks whether credentials exist for the active LLM provider."""
        if self.llm_provider == "gemini":
            return self.has_gemini_credentials
        return bool(self.llm_base_url.strip())


    @property
    def has_telegram_credentials(self) -> bool:
        """Checks whether Telegram Bot token and Chat ID are configured."""
        return bool(
            self.telegram_bot_token.get_secret_value().strip()
            and self.telegram_chat_id.strip()
        )

    @property
    def effective_whatsapp_recipients(self) -> list[str]:
        """Returns consolidated list of WhatsApp recipient JIDs/numbers."""
        recipients = list(self.whatsapp_recipients)
        if self.whatsapp_recipient_id.strip() and self.whatsapp_recipient_id.strip() not in recipients:
            recipients.insert(0, self.whatsapp_recipient_id.strip())
        return recipients

    @property
    def has_whatsapp_credentials(self) -> bool:
        """Checks whether WhatsApp bridge URL and recipient are configured and enabled."""
        return bool(
            self.whatsapp_enabled
            and self.whatsapp_bridge_url.strip()
            and self.effective_whatsapp_recipients
        )

    @property
    def effective_signal_recipients(self) -> list[str]:
        """Returns consolidated list of Signal recipient phone numbers/UUIDs."""
        recipients = list(self.signal_recipients)
        if self.signal_recipient_id.strip() and self.signal_recipient_id.strip() not in recipients:
            recipients.insert(0, self.signal_recipient_id.strip())
        return recipients

    @property
    def has_signal_credentials(self) -> bool:
        """Checks whether Signal REST API, sender number, and recipients are configured and enabled."""
        return bool(
            self.signal_enabled
            and self.signal_bridge_url.strip()
            and self.signal_sender_number.strip()
            and self.effective_signal_recipients
        )

    @property
    def has_tavily_credentials(self) -> bool:
        """Checks whether a valid Tavily API key is configured."""
        return bool(self.tavily_api_key.get_secret_value().strip())

    @property
    def has_exa_credentials(self) -> bool:
        """Checks whether a valid Exa API key is configured."""
        return bool(self.exa_api_key.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the cached global Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
