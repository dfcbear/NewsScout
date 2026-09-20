"""tests/test_config.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for NewsScout configuration management (Pydantic BaseSettings v2),
environment overrides, secret masking, and directory initialization.
"""

from pathlib import Path

from pydantic import SecretStr
import pytest

from newsscout.config import Settings, get_settings


class TestConfigSettings:
    """Verifies Settings configuration, environment overrides, and secret masking."""

    def test_default_values(self):
        s = Settings()
        assert s.app_name == "NewsScout"
        assert s.db_path == Path("data/ai_scout.db")
        assert s.audio_output_dir == Path("data/audio")
        assert s.port == 8000
        assert s.sqlite_busy_timeout_ms == 15000
        assert s.timezone == "Europe/Berlin"
        assert s.schedule_morning == "07:00"
        assert s.schedule_afternoon == "16:00"

    def test_environment_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        custom_db = tmp_path / "custom.db"
        monkeypatch.setenv("DB_PATH", str(custom_db))
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("SCHEDULE_MORNING", "08:30")

        s = Settings()
        assert s.db_path == custom_db
        assert s.log_level == "WARNING"
        assert s.port == 9000
        assert s.schedule_morning == "08:30"

    def test_alias_choices(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        alt_db = tmp_path / "alt_scout.db"
        monkeypatch.setenv("SQLITE_DB_PATH", str(alt_db))
        monkeypatch.setenv("SQLITE_BUSY_TIMEOUT", "8000")
        monkeypatch.setenv("TELEGRAM_TOKEN", "mock_telegram_token_via_alias")

        s = Settings()
        assert s.db_path == alt_db
        assert s.sqlite_busy_timeout_ms == 8000
        assert s.telegram_bot_token.get_secret_value() == "mock_telegram_token_via_alias"

    def test_secret_masking(self, monkeypatch: pytest.MonkeyPatch):
        raw_token = "secret_telegram_bot_token_xyz"
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", raw_token)

        s = Settings()
        assert isinstance(s.telegram_bot_token, SecretStr)
        assert raw_token not in str(s.telegram_bot_token)
        assert raw_token not in repr(s.telegram_bot_token)
        assert s.telegram_bot_token.get_secret_value() == raw_token

    def test_ensure_directories(self, tmp_path: Path):
        s = Settings(
            db_path=tmp_path / "nested" / "dir" / "db.sqlite",
            audio_output_dir=tmp_path / "nested" / "audio",
            preferences_file=tmp_path / "nested" / "prefs" / "pref.json",
        )
        s.ensure_directories()
        assert s.db_path.parent.exists()
        assert s.audio_output_dir.exists()
        assert s.preferences_file.parent.exists()

    def test_credential_check_properties(self):
        s_empty = Settings(gemini_api_key=SecretStr(""), telegram_bot_token=SecretStr(""))
        assert s_empty.has_gemini_credentials is False
        assert s_empty.has_telegram_credentials is False

        s_ready = Settings(
            gemini_api_key=SecretStr("AIzaSyValidGeminiKey"),
            telegram_bot_token=SecretStr("123456:ValidToken"),
            telegram_chat_id="998877",
        )
        assert s_ready.has_gemini_credentials is True
        assert s_ready.has_telegram_credentials is True

    # ------------------------------------------------------------------
    # New validation tests (Paket 4)
    # ------------------------------------------------------------------

    def test_invalid_timezone_raises(self):
        """Invalid IANA timezone should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(timezone="Mars/Olympus")

    def test_valid_timezone_accepted(self):
        """Valid IANA timezone should be accepted."""
        s = Settings(timezone="America/New_York")
        assert s.timezone == "America/New_York"

    def test_invalid_schedule_time_raises(self):
        """Invalid schedule time format should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(schedule_morning="25:00")
        with pytest.raises(ValidationError):
            Settings(schedule_afternoon="12:99")

    def test_pipeline_interval_zero_raises(self):
        """pipeline_interval_hours=0 should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(pipeline_interval_hours=0)

    def test_pipeline_interval_too_large_raises(self):
        """pipeline_interval_hours=169 should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(pipeline_interval_hours=169)

    def test_audio_min_gt_max_raises(self):
        """audio_min > audio_max should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(audio_min_duration_minutes=30, audio_max_duration_minutes=15)

    def test_api_secret_key_default_empty(self):
        """api_secret_key should default to empty (auth disabled)."""
        s = Settings()
        assert s.api_secret_key.get_secret_value() == ""

    def test_api_rate_limit_default(self):
        """api_rate_limit_per_minute should default to 60."""
        s = Settings()
        assert s.api_rate_limit_per_minute == 60

    def test_timeout_ge_constraint(self):
        """Timeouts below 1.0 should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(search_timeout_seconds=0.5)
        with pytest.raises(ValidationError):
            Settings(llm_timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            Settings(delivery_timeout_seconds=0.1)
        with pytest.raises(ValidationError):
            Settings(delivery_gateway_timeout_seconds=0.9)
