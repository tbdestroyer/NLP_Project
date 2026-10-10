"""Loads config.yaml, the single place for settings."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"

# Keep every Hugging Face download (datasets, models) inside the project folder rather than the
# user profile. Must be set before huggingface_hub / transformers are imported.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def load_env() -> None:
    """Reads KEY=VALUE lines from .env into the environment (existing variables win)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry in config `paths`."""
    return ROOT / load_config()["paths"][key]
