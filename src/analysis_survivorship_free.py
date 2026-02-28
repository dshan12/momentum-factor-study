import os
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


def compute_monthly_returns(masked_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Single-month returns requiring both t and t-1 to be valid.
    Hard cap: individual monthly returns clipped to [-40%, +40%].
    S&P 500 constituents very rarely move more than 40% in a single month;
    values beyond that are almost always data artifacts or delistings.
    Then winsorize cross-sectionally at 1/99th pct as an additional pass.
    """
    p1 = masked_prices
    p0 = masked_prices.shift(1)
    rets = (p1 / p0 - 1).where(p0.notna() & p1.notna())

    # Hard cap BEFORE winsorization
    rets = rets.clip(lower=-0.40, upper=0.40)

    def _winsorize(row):
        if row.notna().sum() < 50:
            return row
        lo, hi = row.quantile(0.01), row.quantile(0.99)
        return row.clip(lower=lo, upper=hi)

    return rets.apply(_winsorize, axis=1)


def compute_momentum_ranks(
    masked_prices: pd.DataFrame, lookback: int = 12, skip: int = 1
) -> pd.DataFrame:
    L = np.log(masked_prices)
    mom = (L.shift(skip) - L.shift(lookback + skip)).replace([np.inf, -np.inf], np.nan)
    return mom.rank(axis=1, pct=True, na_option="keep")


def build_signals(ranks, long_q=0.9, short_q=0.1):
    return (ranks >= long_q).astype(int), (ranks <= short_q).astype(int)


def annualized_return(r):
    return (1 + r.mean()) ** 12 - 1


def sharpe_ratio(r, rf=0.02):
    rf_m = (1 + rf) ** (1 / 12) - 1
    ex = r - rf_m
    return (ex.mean() / (ex.std() + 1e-12)) * np.sqrt(12)


def max_drawdown(r):
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    prices = load_prices_union(
        os.path.join(ROOT, "data/cleaned/monthly_adjclose_union.csv")
    )
    prices = prices.where(prices > 0)
    membership = load_membership_monthly(
        os.path.join(ROOT, "data/cleaned/sp500_membership_monthly.csv")
    )
    masked_prices = prices_masked_by_membership(prices, membership)

    print(
        f"[data] panel: {masked_prices.shape}, avg non-null/row: {masked_prices.notna().sum(axis=1).mean():.0f}"
    )

    rets = compute_monthly_returns(masked_prices)
    ranks = compute_momentum_ranks(masked_prices, lookback=12, skip=1)
    longs, shorts = build_signals(ranks)

    valid = (longs.sum(axis=1) >= 20) & (shorts.sum(axis=1) >= 20)
    longs = longs.where(valid, 0)
    shorts = shorts.where(valid, 0)
    print(f"[signals] valid months: {valid.sum()} / {len(valid)}")

    W = equal_weight_long_short(longs, shorts).shift(1)

    gross = (W * rets).sum(axis=1)
    gross = gross.where(W.abs().sum(axis=1) > 0)
    gross.name = "strategy_gross"

    print(
        f"\n[gross] n={gross.notna().sum()}, mean={gross.mean():.4f}, std={gross.std():.4f}"
    )
    print(f"[gross] worst 5:\n{gross.nsmallest(5)}")
    print(f"[gross] best  5:\n{gross.nlargest(5)}")

    # April 2009 debug
    t = pd.Timestamp("2009-04-30")
    if t in W.index and t in rets.index:
        w_t = W.loc[t]
        r_t = rets.loc[t]
        contrib = (w_t * r_t).dropna().sort_values()
        shorts_held = w_t[w_t < -0.001].index
        print(
            f"\n[debug Apr-2009] short returns (winsorized):\n{r_t[shorts_held].sort_values(ascending=False).head(5)}"
        )
        print(f"[debug Apr-2009] sum: {contrib.sum():.4f}")
        print(
            f"[debug Apr-2009] rets max that month: {r_t.max():.4f}, min: {r_t.min():.4f}"
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
    print("\n[TC sensitivity]")
    print(
        sens.to_string(
            index=False,
            formatters={
                "AnnRet": "{:.2%}".format,
                "Vol": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "MaxDD": "{:.2%}".format,
                "AvgTurnover": "{:.2%}".format,
            },
        )
    )

    print("\n=== Survivorship-free (12-1, 10 bps one-way) ===")
    print(f"Ann Return (net): {annualized_return(net):.2%}")
    print(f"Vol: {net.std() * np.sqrt(12):.2%}  Sharpe: {sharpe_ratio(net):.2f}")
    print(f"Max DD: {max_drawdown(net):.2%}")
    print(
        f"Avg turnover: {to.mean():.2%} | Median: {to.median():.2%} | 95th: {to.quantile(0.95):.2%}"
    )

    out_dir = os.path.join(ROOT, "data/cleaned")
    os.makedirs(out_dir, exist_ok=True)
    sens.to_csv(os.path.join(out_dir, "tc_sensitivity.csv"), index=False)
    masked_prices.to_csv(os.path.join(out_dir, "masked_prices_survivorship.csv"))
    rets.to_csv(os.path.join(out_dir, "returns_survivorship.csv"))
    gross.to_csv(os.path.join(out_dir, "strategy_gross_survivorship.csv"))
    net.to_csv(os.path.join(out_dir, "strategy_net_survivorship.csv"))
    to.to_csv(os.path.join(out_dir, "strategy_turnover_survivorship.csv"))
    print("\n[✓] Done.")
    # Check signal direction — do longs actually outperform shorts?
    long_ret = (W.clip(lower=0) * rets).sum(axis=1)
    short_ret = (W.clip(upper=0) * rets).sum(
        axis=1
    )  # will be negative when shorts lose
    print("Long leg ann return:", annualized_return(long_ret.dropna()))
    print("Short leg ann return:", annualized_return(short_ret.dropna()))
    print("Long leg mean monthly:", long_ret.mean())
    print("Short leg mean monthly:", short_ret.mean())
    # Verify: do high-rank stocks (longs) have higher past returns than low-rank (shorts)?
    t_check = pd.Timestamp("2007-06-30")  # a calm pre-crisis month
    print("\nRanks at 2007-06-30 (sample):")
    print(ranks.loc[t_check].dropna().describe())
    long_tickers = longs.loc[t_check][longs.loc[t_check] > 0].index.tolist()[:5]
    short_tickers = shorts.loc[t_check][shorts.loc[t_check] > 0].index.tolist()[:5]
    L = np.log(masked_prices)
    mom_check = (L.shift(1) - L.shift(13)).loc[t_check]
    print(f"\nSample long tickers 12-1 momentum: {mom_check[long_tickers].values}")
    print(f"Sample short tickers 12-1 momentum: {mom_check[short_tickers].values}")


if __name__ == "__main__":
    main()
