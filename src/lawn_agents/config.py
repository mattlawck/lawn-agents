"""Settings loader — merges `config.yaml` and `.env` into a typed object.

`config.yaml` holds non-secret operating configuration (location,
cultivar, thresholds, allowlists). `.env` holds secrets. Both are merged
through Pydantic-Settings and validated at startup; a misconfigured run
dies fast with a clear error.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lawn_agents.models import ChemicalsConfig, WeedsConfig

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CHEMICALS_PATH = Path("data/chemicals.yaml")
DEFAULT_WEEDS_PATH = Path("data/weeds.yaml")

Provider = Literal["gemini", "anthropic"]


class LocationConfig(BaseModel):
    """Geographic and climate-context fields."""

    model_config = ConfigDict(extra="forbid")

    zip: str
    city: str
    state: str
    county_fips: str
    latitude: float
    longitude: float
    coastal: bool = False
    usda_zone: str


class SubjectConfig(BaseModel):
    """Phase-1 single-subject config. Phase 2 will plural-ize into `subjects`."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lawn"]
    cultivar: str
    species: str
    installed_year: int | None = None


class ClimateConfig(BaseModel):
    """Climate-derived thresholds used by the planner."""

    model_config = ConfigDict(extra="forbid")

    last_frost: date
    first_frost: date
    green_up_soil_temp_f: int
    dormancy_soil_temp_f: int
    preemergent_spring_soil_temp_f: int
    preemergent_fall_soil_temp_f: int


class RetrievalConfig(BaseModel):
    """Knobs for hybrid retrieval + the weak-result threshold."""

    model_config = ConfigDict(extra="forbid")

    top_k_vector: int = 8
    top_k_bm25: int = 4
    rerank_top_k: int = 5
    # Tiered relevance check (PR-tba):
    #   score < weak    → weak (research subagent fires)
    #   score >= strong → strong (skip extra checks)
    #   between         → run lexical-overlap check; on miss, escalate
    #                     to an LLM relevance gate via the cheap router
    #                     model. Catches semantic mismatches like the
    #                     Japanese-clover→sedge probe in PR #32.
    weak_score_threshold: float = 0.55
    strong_score_threshold: float = 0.70


class KnowledgeConfig(BaseModel):
    """RAG layer configuration."""

    model_config = ConfigDict(extra="forbid")

    corpus_dir: Path
    index_dir: Path
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class ResearchConfig(BaseModel):
    """Self-extending RAG (ADR 0005) configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    domain_allowlist: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    """Provider selection plus model IDs for routing and synthesis.

    `provider` picks which adapter `build_chat_model` constructs; the
    corresponding `*_API_KEY` must be present in `.env`. See ADR 0006.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider = "gemini"
    router: str
    synthesizer: str


class HttpConfig(BaseModel):
    """Shared HTTP-client knobs."""

    model_config = ConfigDict(extra="forbid")

    contact_email: EmailStr
    timeout_seconds: float = 15.0
    retries: int = 3


SinkName = Literal["console", "email", "sms"]


def _default_sinks() -> list[SinkName]:
    return ["console"]


class NotifyConfig(BaseModel):
    """Output sinks for the rendered Recommendation."""

    model_config = ConfigDict(extra="forbid")

    sinks: list[SinkName] = Field(default_factory=_default_sinks)


class AppConfig(BaseModel):
    """The non-secret half of settings, loaded from `config.yaml`."""

    model_config = ConfigDict(extra="forbid")

    location: LocationConfig
    subject: SubjectConfig
    climate: ClimateConfig
    knowledge: KnowledgeConfig
    research: ResearchConfig
    seed_urls: list[str] = Field(default_factory=list)
    chemicals_file: Path = Field(default=DEFAULT_CHEMICALS_PATH)
    weeds_file: Path = Field(default=DEFAULT_WEEDS_PATH)
    models: ModelsConfig
    http: HttpConfig
    notify: NotifyConfig


class Settings(BaseSettings):
    """Top-level settings: secrets from `.env`, app config from `config.yaml`.

    Both provider keys are optional at the env layer; a model validator
    enforces that whichever provider is selected in
    `config.yaml > models.provider` has its key present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    nws_user_agent: str | None = Field(default=None, alias="NWS_USER_AGENT")
    lawn_agents_index_dir: Path | None = Field(default=None, alias="LAWN_AGENTS_INDEX_DIR")

    app: AppConfig
    chemicals: ChemicalsConfig = Field(default_factory=ChemicalsConfig)
    weeds: WeedsConfig = Field(default_factory=WeedsConfig)

    @model_validator(mode="after")
    def _require_key_for_provider(self) -> Self:
        provider = self.app.models.provider
        key = self.gemini_api_key if provider == "gemini" else self.anthropic_api_key
        if key is None:
            env_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
            msg = (
                f"models.provider={provider!r} but {env_name} is unset. "
                "Add it to .env (see .env.example)."
            )
            raise ValueError(msg)
        return self

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
        """Load and validate settings from `.env` and the given YAML file.

        Args:
            config_path: Path to `config.yaml`. Defaults to the project root.

        Returns:
            A fully validated `Settings` instance.

        Raises:
            FileNotFoundError: If `config_path` does not exist.
            pydantic.ValidationError: If either source fails validation.
        """
        # Export `.env` to `os.environ` so libraries that read process
        # env vars directly (e.g., huggingface_hub via HF_TOKEN) pick up
        # the values. pydantic-settings reads `.env` separately into our
        # typed fields below; this is complementary, not redundant.
        #
        # We pass an explicit CWD-relative path because `load_dotenv()`
        # with no args calls `find_dotenv()`, which walks up from the
        # *calling file's directory* — so it finds the project `.env`
        # even when CWD is a test tmp path. With an explicit relative
        # path the CWD is respected, matching pydantic-settings'
        # behavior.
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"))

        if not config_path.exists():
            msg = f"config file not found: {config_path}"
            raise FileNotFoundError(msg)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        app = AppConfig.model_validate(raw)
        chemicals = ChemicalsConfig()
        if app.chemicals_file.exists():
            chemicals_raw = yaml.safe_load(app.chemicals_file.read_text(encoding="utf-8")) or {}
            chemicals = ChemicalsConfig.model_validate(chemicals_raw)
        weeds = WeedsConfig()
        if app.weeds_file.exists():
            weeds_raw = yaml.safe_load(app.weeds_file.read_text(encoding="utf-8")) or {}
            weeds = WeedsConfig.model_validate(weeds_raw)
        return cls(app=app, chemicals=chemicals, weeds=weeds)
