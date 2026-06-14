from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_prices_union(
    path: str = "data/cleaned/monthly_adjclose_union.csv",
) -> pd.DataFrame:
    """Load the union of monthly adjusted close prices for all tickers.

    Args:
        path: Path to the CSV file containing monthly adjusted close prices.

    Returns:
        DataFrame with dates as index (month-end normalized) and tickers as columns.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")
    # Snap price index to month-end to guarantee alignment with membership
    df.index = df.index + pd.offsets.MonthEnd(0)
    df.index = pd.DatetimeIndex(df.index).normalize() + pd.offsets.MonthEnd(0)
    df.columns = [str(c) for c in df.columns]
    return df


def load_membership_monthly(
    path: str = "data/cleaned/sp500_membership_monthly.csv",
) -> pd.DataFrame:
    """Load the monthly S&P 500 membership panel.

    Args:
        path: Path to the CSV file with columns date, ticker, in_index.

    Returns:
        DataFrame with columns date (month-end normalized), ticker, in_index.
    """
    m = pd.read_csv(path, parse_dates=["date"])
    m["ticker"] = m["ticker"].astype(str)
    # Snap membership dates to month-end as well
    m["date"] = pd.to_datetime(m["date"]) + pd.offsets.MonthEnd(0)
    return m


def prices_masked_by_membership(
    prices: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Mask price DataFrame to keep only prices when a stock was in the index.

    Args:
        prices: DataFrame with dates as index and tickers as columns.
        membership: DataFrame with columns date, ticker, in_index.

    Returns:
        Price DataFrame with NaN for periods when a stock was not in the index.

    Raises:
        ValueError: If no overlapping tickers exist between prices and membership.
    """
    mask = (
        membership.pivot_table(
            index="date", columns="ticker", values="in_index", aggfunc="max"
        )
        .reindex(prices.index)
        .fillna(0.0)
    )

    # Sanity check: warn if mask coverage looks too low
    coverage = (mask > 0.5).sum(axis=1)
    if coverage.mean() < 50:
        warnings.warn(
            f"Low average membership coverage: {coverage.mean():.1f} stocks/month. "
            "Check date alignment between prices and membership CSV.",
            RuntimeWarning,
            stacklevel=2,
        )

    common = sorted(set(prices.columns).intersection(set(mask.columns)))
    if not common:
        raise ValueError(
            "No overlapping tickers between prices and membership. "
            "Check that ticker formats match (e.g. BRK-B vs BRK.B)."
        )

    p = prices[common].copy()
    m = mask[common].astype(float)
    p = p.where(m > 0.5, np.nan)
    return p
