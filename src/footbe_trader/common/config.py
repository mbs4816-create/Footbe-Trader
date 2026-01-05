"""Application configuration loading and models."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = "data/footbe.db"
    echo: bool = False


class FootballApiConfig(BaseModel):
    """API-Football configuration."""

    base_url: str = "https://v3.football.api-sports.io"
    api_key: str = ""
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 30


class KalshiConfig(BaseModel):
    """Kalshi API configuration."""

    # API endpoints - Kalshi moved sports/elections to new domain
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    demo_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    use_demo: bool = False  # Use production API (has real soccer markets)

    # Authentication
    api_key_id: str = ""  # Your API key ID (UUID)
    private_key_path: str = ""  # Path to private key .pem file
    private_key: str = ""  # Or private key content directly

    # Request settings
    timeout_seconds: int = 30
    rate_limit_per_second: float = 10.0  # Kalshi rate limit
    max_retries: int = 3
    retry_backoff_base: float = 1.0  # Base seconds for exponential backoff

    @property
    def effective_base_url(self) -> str:
        """Get the effective base URL based on demo mode."""
        return self.demo_base_url if self.use_demo else self.base_url

    def get_private_key_content(self) -> str:
        """Get private key content from file or direct config.

        Returns:
            Private key PEM content.

        Raises:
            ValueError: If no private key is configured.
        """
        if self.private_key:
            # Handle escaped newlines from env vars
            return self.private_key.replace("\\n", "\n")

        if self.private_key_path:
            key_path = Path(self.private_key_path).expanduser()
            if key_path.exists():
                return key_path.read_text()
            raise ValueError(f"Private key file not found: {key_path}")

        raise ValueError(
            "No Kalshi private key configured. "
            "Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY in .env"
        )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # "json" or "console"
    log_file: str | None = None


class AgentConfig(BaseModel):
    """Agent/heartbeat configuration."""

    name: str = "footbe-agent"
    heartbeat_interval_seconds: int = 60
    dry_run: bool = True


class AppConfig(BaseModel):
    """Root application configuration."""

    environment: str = Field(default="dev", description="Environment name (dev/prod)")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    football_api: FootballApiConfig = Field(default_factory=FootballApiConfig)
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: str | Path) -> AppConfig:
    """Load configuration from a YAML file with environment variable substitution.

    Environment variables can override config values. The following env vars are checked:
    - KALSHI_API_KEY_ID: Kalshi API key ID
    - KALSHI_PRIVATE_KEY_PATH: Path to private key file
    - KALSHI_PRIVATE_KEY: Private key content directly
    - KALSHI_ENVIRONMENT: "demo" or "production"
    - FOOTBALL_API_KEY: Football API key

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed AppConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is invalid.
    """
    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, rely on system env vars

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        raw_config: dict[str, Any] = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    _apply_env_overrides(raw_config)

    return AppConfig(**raw_config)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply environment variable overrides to config dict.

    Args:
        config: Configuration dictionary to modify in place.
    """
    # Ensure nested dicts exist
    if "kalshi" not in config:
        config["kalshi"] = {}
    if "football_api" not in config:
        config["football_api"] = {}

    # Kalshi credentials from env
    if api_key_id := os.environ.get("KALSHI_API_KEY_ID"):
        config["kalshi"]["api_key_id"] = api_key_id

    if private_key_path := os.environ.get("KALSHI_PRIVATE_KEY_PATH"):
        config["kalshi"]["private_key_path"] = private_key_path

    if private_key := os.environ.get("KALSHI_PRIVATE_KEY"):
        config["kalshi"]["private_key"] = private_key

    if kalshi_env := os.environ.get("KALSHI_ENVIRONMENT"):
        config["kalshi"]["use_demo"] = kalshi_env.lower() != "production"

    # Football API from env
    if football_key := os.environ.get("FOOTBALL_API_KEY"):
        config["football_api"]["api_key"] = football_key
