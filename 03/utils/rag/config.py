from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path, so notebooks in any subfolder of 03/ find the same file.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """CLARIN API settings read from ``03/.env`` (prefix ``CLARIN_``)."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="CLARIN_", extra="ignore")

    api_key: SecretStr
    base_url: str = "https://services.clarin-pl.eu/api/v1/oapi"
    model: str = ""


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process.

    Returns:
        Validated settings.

    Raises:
        pydantic.ValidationError: When ``CLARIN_API_KEY`` is missing.
    """
    return Settings()  # type: ignore[call-arg]  # api_key comes from the env file
