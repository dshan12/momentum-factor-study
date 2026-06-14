from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data.panel_from_membership import (
    load_prices_union,
    load_membership_monthly,
    prices_masked_by_membership,
)
from data.turnover import (
    equal_weight_long_short,
    turnover_from_weights,
    apply_turnover_costs,
)

logger = logging.getLogger(__name__)


def compute_monthly_returns(masked_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute single-month returns with hard cap and cross-sectional winsorization.

    Requires both t and t-1 prices to be valid. Individual monthly returns are
    hard-clipped to [-40%, +40%]; S&P 500 constituents rarely move more than
    40% in a single month — values beyond that are almost always data artifacts
    or delistings. An additional cross-sectional winsorization at the 1st/99th
    percentile is applied.

    Args:
        masked_prices: DataFrame of month-end adjusted close prices masked by
            S&P 500 membership, shape (n_months, n_tickers).

    Returns:
        DataFrame of monthly returns, same shape as input.
    """
    p1 = masked_prices
    p0 = masked_prices.shift(1)
    rets = (p1 / p0 - 1).where(p0.notna() & p1.notna())

    rets = rets.clip(lower=-0.40, upper=0.40)

    def _winsorize(row: pd.Series) -> pd.Series:
        if row.notna().sum() < 50:
            return row
        lo, hi = row.quantile(0.01), row.quantile(0.99)
        return row.clip(lower=lo, upper=hi)

    return rets.apply(_winsorize, axis=1)


def compute_momentum_ranks(
    masked_prices: pd.DataFrame, lookback: int = 12, skip: int = 1
) -> pd.DataFrame:
    """Compute cross-sectional percentile ranks for 12-1 month momentum.

    Args:
        masked_prices: DataFrame of month-end adjusted close prices.
        lookback: Number of months for the formation period (default 12).
        skip: Number of months to skip before the current (default 1).

    Returns:
        DataFrame of percentile ranks (0-1) per period, higher = stronger
        past return.
    """
    L = np.log(masked_prices)
    mom = (L.shift(skip) - L.shift(lookback + skip)).replace([np.inf, -np.inf], np.nan)
    return mom.rank(axis=1, pct=True, na_option="keep")


def build_signals(
    ranks: pd.DataFrame, long_q: float = 0.9, short_q: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate long/short indicator signals from momentum percentile ranks.

    Args:
        ranks: DataFrame of momentum percentile ranks (0-1).
        long_q: Quantile threshold above which stocks are classified as long
            (default 0.9).
        short_q: Quantile threshold below which stocks are classified as short
            (default 0.1).

    Returns:
        Tuple of (longs, shorts) DataFrames with boolean-like integer signals.
    """
    return (ranks >= long_q).astype(int), (ranks <= short_q).astype(int)


def annualized_return(r: pd.Series) -> float:
    """Annualize a mean monthly return.

    Args:
        r: Series of monthly returns (decimals).

    Returns:
        Annualized return as a decimal.
    """
    return (1 + r.mean()) ** 12 - 1


def sharpe_ratio(r: pd.Series, rf: float = 0.02) -> float:
    """Compute annualized Sharpe ratio from monthly return series.

    Args:
        r: Series of monthly returns (decimals).
        rf: Annual risk-free rate (default 0.02 for 2%).

    Returns:
        Annualized Sharpe ratio.
    """
    rf_m = (1 + rf) ** (1 / 12) - 1
    ex = r - rf_m
    return (ex.mean() / (ex.std() + 1e-12)) * np.sqrt(12)


def max_drawdown(r: pd.Series) -> float:
    """Compute the maximum drawdown from a series of returns.

    Args:
        r: Series of monthly returns (decimals).

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.50 for -50%).
    """
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def main() -> None:
    """Run the full survivorship-free momentum analysis pipeline."""
    ROOT = Path(__file__).resolve().parent.parent

    prices = load_prices_union(
        ROOT / "data/cleaned/monthly_adjclose_union.csv"
    )
    prices = prices.where(prices > 0)
    membership = load_membership_monthly(
        ROOT / "data/cleaned/sp500_membership_monthly.csv"
    )
    masked_prices = prices_masked_by_membership(prices, membership)

    logger.info(
        "data panel: %s, avg non-null/row: %.0f",
        masked_prices.shape,
        masked_prices.notna().sum(axis=1).mean(),
    )

    rets = compute_monthly_returns(masked_prices)
    ranks = compute_momentum_ranks(masked_prices, lookback=12, skip=1)
    longs, shorts = build_signals(ranks)

    valid = (longs.sum(axis=1) >= 20) & (shorts.sum(axis=1) >= 20)
    longs = longs.where(valid, 0)
    shorts = shorts.where(valid, 0)
    logger.info("signals valid months: %d / %d", valid.sum(), len(valid))

    W = equal_weight_long_short(longs, shorts).shift(1)

    gross = (W * rets).sum(axis=1)
    gross = gross.where(W.abs().sum(axis=1) > 0)
    gross.name = "strategy_gross"

    logger.info(
        "gross n=%d, mean=%.4f, std=%.4f",
        gross.notna().sum(),
        gross.mean(),
        gross.std(),
    )
    logger.info("gross worst 5:\n%s", gross.nsmallest(5))
    logger.info("gross best  5:\n%s", gross.nlargest(5))

    t = pd.Timestamp("2009-04-30")
    if t in W.index and t in rets.index:
        w_t = W.loc[t]
        r_t = rets.loc[t]
        contrib = (w_t * r_t).dropna().sort_values()
        shorts_held = w_t[w_t < -0.001].index
        logger.debug(
            "debug Apr-2009 short returns (winsorized):\n%s",
            r_t[shorts_held].sort_values(ascending=False).head(5),
        )
        logger.debug("debug Apr-2009 sum: %.4f", contrib.sum())
        logger.debug(
            "debug Apr-2009 rets max that month: %.4f, min: %.4f",
            r_t.max(),
            r_t.min(),
        )

    to = turnover_from_weights(W, rets)
    COST_BPS = 10
    net = apply_turnover_costs(gross, to, COST_BPS).dropna()

    rows = []
    for bps in [5, 10, 15, 25]:
        r = (gross - (bps / 1e4) * to).dropna()
        rows.append(
            {
                "Cost_bps": bps,
                "AnnRet": annualized_return(r),
                "Vol": r.std() * np.sqrt(12),
                "Sharpe": sharpe_ratio(r),
                "MaxDD": max_drawdown(r),
                "AvgTurnover": to.mean(),
            }
        )
    sens = pd.DataFrame(rows)
    logger.info(
        "TC sensitivity:\n%s",
        sens.to_string(
            index=False,
            formatters={
                "AnnRet": "{:.2%}".format,
                "Vol": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "MaxDD": "{:.2%}".format,
                "AvgTurnover": "{:.2%}".format,
            },
        ),
    )

    logger.info("=== Survivorship-free (12-1, 10 bps one-way) ===")
    logger.info("Ann Return (net): %.2f%%", annualized_return(net) * 100)
    logger.info(
        "Vol: %.2f%%  Sharpe: %.2f",
        net.std() * np.sqrt(12) * 100,
        sharpe_ratio(net),
    )
    logger.info("Max DD: %.2f%%", max_drawdown(net) * 100)
    logger.info(
        "Avg turnover: %.2f%% | Median: %.2f%% | 95th: %.2f%%",
        to.mean() * 100,
        to.median() * 100,
        to.quantile(0.95) * 100,
    )

    out_dir = ROOT / "data/cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    sens.to_csv(out_dir / "tc_sensitivity.csv", index=False)
    masked_prices.to_csv(out_dir / "masked_prices_survivorship.csv")
    rets.to_csv(out_dir / "returns_survivorship.csv")
    gross.to_csv(out_dir / "strategy_gross_survivorship.csv")
    net.to_csv(out_dir / "strategy_net_survivorship.csv")
    to.to_csv(out_dir / "strategy_turnover_survivorship.csv")
    logger.info("Done.")

    long_ret = (W.clip(lower=0) * rets).sum(axis=1)
    short_ret = (W.clip(upper=0) * rets).sum(axis=1)
    logger.info("Long leg ann return: %.4f", annualized_return(long_ret.dropna()))
    logger.info("Short leg ann return: %.4f", annualized_return(short_ret.dropna()))
    logger.info("Long leg mean monthly: %.6f", long_ret.mean())
    logger.info("Short leg mean monthly: %.6f", short_ret.mean())

    t_check = pd.Timestamp("2007-06-30")
    logger.info(
        "Ranks at 2007-06-30 (sample):\n%s",
        ranks.loc[t_check].dropna().describe(),
    )
    long_tickers = longs.loc[t_check][longs.loc[t_check] > 0].index.tolist()[:5]
    short_tickers = shorts.loc[t_check][shorts.loc[t_check] > 0].index.tolist()[:5]
    L = np.log(masked_prices)
    mom_check = (L.shift(1) - L.shift(13)).loc[t_check]
    logger.info(
        "Sample long tickers 12-1 momentum: %s", mom_check[long_tickers].values
    )
    logger.info(
        "Sample short tickers 12-1 momentum: %s", mom_check[short_tickers].values
    )


if __name__ == "__main__":
    main()
