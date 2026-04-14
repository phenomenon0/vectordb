"""Configuration management — TOML-based, XDG-friendly."""

import tomllib
from pathlib import Path
from dataclasses import dataclass, field

CONFIG_DIR = Path.home() / ".config" / "jobhunter"
CONFIG_FILE = CONFIG_DIR / "config.toml"
DATA_DIR = Path.home() / ".local" / "share" / "jobhunter"
DB_PATH = DATA_DIR / "jobs.db"
RESUMES_DIR = DATA_DIR / "resumes"

DEFAULT_CONFIG = """\
# JobHunter configuration

[llm]
# Anthropic model ID
# Options: "claude-haiku-4-5-20251001" (fast/cheap), "claude-sonnet-4-20250514" (balanced)
model = "claude-haiku-4-5-20251001"

# Minimum fit score (1-10) to consider a job relevant
min_score = 6

[scrape]
# Default sites to scrape: linkedin, indeed, glassdoor, zip_recruiter, google
sites = ["linkedin", "indeed"]

# Default number of results per site
results_wanted = 50

# Country for job search
country = "USA"

[resume]
# Path to your base resume (PDF, TXT, or MD)
# path = "/home/you/resume.pdf"
"""


@dataclass
class LLMConfig:
    model: str = "claude-haiku-4-5-20251001"
    min_score: int = 6


@dataclass
class ScrapeConfig:
    sites: list[str] = field(default_factory=lambda: ["linkedin", "indeed"])
    results_wanted: int = 50
    country: str = "USA"


@dataclass
class ResumeConfig:
    path: str | None = None


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    scrape: ScrapeConfig = field(default_factory=ScrapeConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


def ensure_dirs():
    """Create config and data directories if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)


def init_config():
    """Write default config if it doesn't exist."""
    ensure_dirs()
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(DEFAULT_CONFIG)


def load_config() -> Config:
    """Load config from TOML file, falling back to defaults."""
    ensure_dirs()
    cfg = Config()

    if CONFIG_FILE.exists():
        raw = tomllib.loads(CONFIG_FILE.read_text())

        if "llm" in raw:
            cfg.llm = LLMConfig(
                model=raw["llm"].get("model", cfg.llm.model),
                min_score=raw["llm"].get("min_score", cfg.llm.min_score),
            )
        if "scrape" in raw:
            cfg.scrape = ScrapeConfig(
                sites=raw["scrape"].get("sites", cfg.scrape.sites),
                results_wanted=raw["scrape"].get("results_wanted", cfg.scrape.results_wanted),
                country=raw["scrape"].get("country", cfg.scrape.country),
            )
        if "resume" in raw:
            cfg.resume = ResumeConfig(
                path=raw["resume"].get("path"),
            )

    return cfg
