import pandas as pd
import pytest

from inclination.config import load_config, path


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def processed():
    """(turns, conversations) from data/processed; tests using it are skipped before `make data`."""
    folder = path("processed")
    if not (folder / "turns.parquet").exists():
        pytest.skip("data/processed is empty; run `make data` first")
    return (
        pd.read_parquet(folder / "turns.parquet"),
        pd.read_parquet(folder / "conversations.parquet"),
    )
