import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_returns() -> pd.Series:
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.01, 0.04, 60), index=idx, name="strategy")


@pytest.fixture
def sample_factor_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.008, 0.04, 60),
            "SMB": rng.normal(0.002, 0.02, 60),
            "HML": rng.normal(0.001, 0.02, 60),
            "RMW": rng.normal(0.003, 0.015, 60),
            "CMA": rng.normal(0.001, 0.012, 60),
            "UMD": rng.normal(0.004, 0.03, 60),
            "RF": np.full(60, 0.00167),
        },
        index=idx,
    )


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(42)
    base = 100.0
    prices = {f"TICK{i}": base * (1 + rng.normal(0.01, 0.04, 12)).cumprod()
              for i in range(5)}
    return pd.DataFrame(prices, index=idx)


@pytest.fixture
def sample_membership_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    rows = []
    for d in idx:
        for i in range(5):
            rows.append({"date": d, "ticker": f"TICK{i}", "in_index": 1})
    return pd.DataFrame(rows)
