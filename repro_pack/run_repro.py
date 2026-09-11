"""Reproduce the 12-1 S&P 500 momentum result on frozen inputs.

Reads pre-computed series from ../data/cleaned/, applies fixed
in-sample / validation / out-of-sample time splits, compares the
long-short strategy against an equal-risk market baseline, and writes
tables and figures to repro_pack/outputs/.

Usage:
    python run_repro.py [--data-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "cleaned"
DEFAULT_OUT = HERE / "outputs"

IS_END = pd.Timestamp("2014-12-31")
VAL_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-31")
RF_ANNUAL = 0.02
HAC_LAGS = 6
COST_BPS_BASE = 10


def load_series(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    s = df[col].copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().asfreq("ME")


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r.mean()) ** 12 - 1.0)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(12.0))


def sharpe(r: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    r = r.dropna()
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    ex = r - rf_m
    return float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(12.0))


def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    w = (1.0 + r).cumprod()
    return float((w / w.cummax() - 1.0).min())


def hac_mean_tstat(r: pd.Series, rf_annual: float = RF_ANNUAL) -> tuple[float, float]:
    """t-statistic and two-sided p-value for mean excess return, HAC(6)."""
    r = r.dropna()
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    y = (r - rf_m).values
    X = np.ones((len(y), 1))
    res = sm.OLS(y, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    return float(res.tvalues[0]), float(res.pvalues[0])


def split(s: pd.Series) -> dict[str, pd.Series]:
    return {
        "IS (2006-2014)": s[s.index <= IS_END],
        "Validation (2015-2019)": s[(s.index > IS_END) & (s.index <= VAL_END)],
        "Test/OOS (2020-2024)": s[s.index >= TEST_START],
        "Full (2006-2024)": s,
    }


def summary_frame(splits: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, r in splits.items():
        r = r.dropna()
        t, p = hac_mean_tstat(r) if len(r) > 12 else (np.nan, np.nan)
        rows.append(
            {
                "Split": label,
                "N": int(len(r)),
                "AnnRet": ann_return(r),
                "AnnVol": ann_vol(r),
                "Sharpe": sharpe(r),
                "MaxDD": max_drawdown(r),
                "HAC-t": t,
                "HAC-p": p,
            }
        )
    return pd.DataFrame(rows).set_index("Split")


def ff_alpha_on_test(
    net: pd.Series, factors: pd.DataFrame
) -> dict[str, float | int]:
    test = net[net.index >= TEST_START].dropna()
    fac = factors.loc[test.index]
    y = test - fac["RF"]
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    X = sm.add_constant(fac[cols])
    res = sm.OLS(y, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    alpha_m = float(res.params["const"])
    return {
        "alpha_ann": float((1.0 + alpha_m) ** 12 - 1.0),
        "alpha_t": float(res.tvalues["const"]),
        "alpha_p": float(res.pvalues["const"]),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
    }


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    gross = load_series(data / "strategy_gross_survivorship.csv", "strategy_gross")
    net = load_series(data / "strategy_net_survivorship.csv", "strategy_net")
    turnover = load_series(data / "strategy_turnover_survivorship.csv", "turnover")
    factors = (
        pd.read_csv(data / "ff5_umd_monthly.csv", parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    market = (factors["Mkt-RF"] + factors["RF"]).rename("us_market").asfreq("ME")

    idx = net.index.intersection(market.index)
    net_a, mkt_a = net.loc[idx], market.loc[idx]

    strat_tbl = summary_frame(split(net_a))
    mkt_tbl = summary_frame(split(mkt_a))
    strat_tbl.to_csv(out / "strategy_by_split.csv")
    mkt_tbl.to_csv(out / "baseline_market_by_split.csv")

    # Transaction-cost sensitivity on the test window.
    g, to = gross.align(turnover, join="inner")
    tc_rows = []
    for bps in [5, 10, 15, 25]:
        r = (g - bps / 1e4 * to).dropna()
        r = r[r.index >= TEST_START]
        tc_rows.append(
            {
                "Cost_bps": bps,
                "AnnRet": ann_return(r),
                "AnnVol": ann_vol(r),
                "Sharpe": sharpe(r),
                "MaxDD": max_drawdown(r),
            }
        )
    tc_tbl = pd.DataFrame(tc_rows).set_index("Cost_bps")
    tc_tbl.to_csv(out / "tc_sensitivity_test.csv")

    alpha = ff_alpha_on_test(net_a, factors)
    pd.DataFrame([alpha]).to_csv(out / "ff5_umd_alpha_test.csv", index=False)

    # Figures.
    wealth = pd.DataFrame(
        {
            "Strategy (net, 10 bps)": (1 + net_a).cumprod(),
            "US Market": (1 + mkt_a).cumprod(),
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth.plot(ax=ax)
    ax.axvline(TEST_START, color="k", linestyle="--", linewidth=1)
    ax.text(TEST_START, ax.get_ylim()[1], "  OOS start", va="top", fontsize=8)
    ax.set_title("Cumulative wealth (log scale not applied; base = 1.0)")
    ax.set_ylabel("Wealth")
    fig.tight_layout()
    fig.savefig(out / "equity_by_split.png", dpi=200)
    plt.close(fig)

    dd = wealth / wealth.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 4))
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(out / "drawdown.png", dpi=200)
    plt.close(fig)

    with open(out / "verification.log", "w") as f:
        for name in [
            "strategy_net_survivorship.csv",
            "strategy_gross_survivorship.csv",
            "strategy_turnover_survivorship.csv",
            "ff5_umd_monthly.csv",
        ]:
            f.write(f"{name} sha256[:16]={sha16(data / name)}\n")
        f.write(f"test N(strategy)={int(net_a[net_a.index >= TEST_START].dropna().shape[0])}\n")
        f.write(
            "test AnnRet={:.4f} Sharpe={:.3f} MaxDD={:.4f}\n".format(
                strat_tbl.loc["Test/OOS (2020-2024)", "AnnRet"],
                strat_tbl.loc["Test/OOS (2020-2024)", "Sharpe"],
                strat_tbl.loc["Test/OOS (2020-2024)", "MaxDD"],
            )
        )
        f.write(
            "FF5+UMD test alpha_ann={:.4f} t={:.2f} p={:.4f} R2={:.3f} n={:d}\n".format(
                alpha["alpha_ann"],
                alpha["alpha_t"],
                alpha["alpha_p"],
                alpha["r2"],
                alpha["n"],
            )
        )

    print(strat_tbl.round(4).to_string())
    print()
    print("Transaction-cost sensitivity (test window):")
    print(tc_tbl.round(4).to_string())
    print()
    print(
        "FF5+UMD alpha (test): ann={:.2%} t={:.2f} p={:.4f} R2={:.3f} n={:d}".format(
            alpha["alpha_ann"], alpha["alpha_t"], alpha["alpha_p"],
            alpha["r2"], alpha["n"],
        )
    )
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
