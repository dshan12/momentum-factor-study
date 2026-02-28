import pandas as pd
import numpy as np


def load_prices_union(path="data/cleaned/monthly_adjclose_union.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")
    # Snap price index to month-end to guarantee alignment with membership
    df.index = df.index + pd.offsets.MonthEnd(0)
    df.index = pd.DatetimeIndex(df.index).normalize() + pd.offsets.MonthEnd(0)
    df.columns = [str(c) for c in df.columns]
    return df


def load_membership_monthly(
    path="data/cleaned/sp500_membership_monthly.csv",
) -> pd.DataFrame:
    m = pd.read_csv(path, parse_dates=["date"])
    m["ticker"] = m["ticker"].astype(str)
    # Snap membership dates to month-end as well
    m["date"] = pd.to_datetime(m["date"]) + pd.offsets.MonthEnd(0)
    return m


def prices_masked_by_membership(
    prices: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    mask = (
        membership.pivot_table(
            index="date", columns="ticker", values="in_index", aggfunc="max"
        )
        .reindex(prices.index)  # now safe: both are month-end
        .fillna(0.0)  # NaN means "not in index that month" → mask out
    )

    # Sanity check: warn if mask coverage looks too low
    coverage = (mask > 0.5).sum(axis=1)
    if coverage.mean() < 50:
        import warnings

        warnings.warn(
            f"Low average membership coverage: {coverage.mean():.1f} stocks/month. "
            "Check date alignment between prices and membership CSV.",
            RuntimeWarning,
        )

    common = sorted(set(prices.columns).intersection(set(mask.columns)))
    if not common:
        raise ValueError(
            "No overlapping tickers between prices and membership. "
            "Check that ticker formats match (e.g. BRK-B vs BRK.B)."
        )

    P = prices[common].copy()
    M = mask[common].astype(float)
    P = P.where(M > 0.5, np.nan)
    return P
