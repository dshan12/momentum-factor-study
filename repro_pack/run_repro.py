"""Reproduce the 12-1 S&P 500 momentum result on frozen inputs.

Reads pre-computed series from ../data/cleaned/, applies fixed
in-sample / validation / out-of-sample time splits, compares the
long-short strategy against a market total-return baseline, and writes
tables and figures to repro_pack/outputs/.

Includes a raw-return sensitivity check that rebuilds the strategy from
the masked price panel without the +/-40% cap and without
cross-sectional winsorization, isolating the effect of those
transformations on the locked test window.

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


def raw_return_panels(masked_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build raw, capped-only, and final return panels from masked prices.

    Mirrors src/analysis_survivorship_free.compute_monthly_returns but
    exposes each transformation stage so the cap + winsorization effect
    can be isolated. Signal ranks are unaffected (built from prices).
    """
    p1 = masked_prices
    p0 = masked_prices.shift(1)
    raw = (p1 / p0 - 1).where(p0.notna() & p1.notna())
    capped = raw.clip(lower=-0.40, upper=0.40)

    def _winsorize(row: pd.Series) -> pd.Series:
        if row.notna().sum() < 50:
            return row
        lo, hi = row.quantile(0.01), row.quantile(0.99)
        return row.clip(lower=lo, upper=hi)

    final = capped.apply(_winsorize, axis=1)
    return {"raw": raw, "capped": capped, "final": final}


def momentum_weights(masked_prices: pd.DataFrame) -> pd.DataFrame:
    """Rebuild execution weights with the locked 12-1 rule.

    Ranks on log(P_{t-1}) - log(P_{t-13}), top/bottom deciles,
    min 20 stocks per leg, equal-weight dollar-neutral, shifted one
    month before trading. Identical rule for all return panels.
    """
    import numpy as np

    L = np.log(masked_prices.where(masked_prices > 0))
    mom = (L.shift(1) - L.shift(13)).replace([np.inf, -np.inf], np.nan)
    ranks = mom.rank(axis=1, pct=True, na_option="keep")
    longs = (ranks >= 0.9).astype(int)
    shorts = (ranks <= 0.1).astype(int)
    valid = (longs.sum(axis=1) >= 20) & (shorts.sum(axis=1) >= 20)
    longs = longs.where(valid, 0)
    shorts = shorts.where(valid, 0)
    lm = (longs > 0).astype(float)
    sm_ = (shorts > 0).astype(float)
    lw = lm.div(lm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = -sm_.div(sm_.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = lw.add(sw, fill_value=0.0).shift(1)
    return W


def drift_weights(w_prev: pd.Series, r_prev: pd.Series) -> pd.Series:
    cols = w_prev.index.union(r_prev.index)
    w = w_prev.reindex(cols).fillna(0.0)
    r = r_prev.reindex(cols).fillna(0.0)
    wd = w * (1.0 + r)
    lp, sp = wd > 0, wd < 0
    ls, ss = wd[lp].sum(), wd[sp].abs().sum()
    out = pd.Series(0.0, index=cols)
    if ls > 1e-12:
        out[lp] = wd[lp] / ls
    if ss > 1e-12:
        out[sp] = wd[sp] / ss
    return out


def turnover_from_weights(W: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    to: list[float] = []
    for i in range(len(W)):
        if i == 0:
            to.append(0.0)
            continue
        w_pre = drift_weights(W.iloc[i - 1], rets.iloc[i - 1])
        w_t = W.iloc[i].reindex(W.columns).fillna(0.0)
        w_pre = w_pre.reindex(W.columns).fillna(0.0)
        to.append(0.5 * (w_t - w_pre).abs().sum())
    return pd.Series(to, index=W.index, name="turnover")


def raw_sensitivity(
    data: Path, cost_bps: float = COST_BPS_BASE
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rebuild strategy net returns under raw / capped / final panels.

    Returns a per-split comparison table (test window is locked) and a
    dict of panel-level diagnostics (cap hit rate, worst-month deltas).
    """
    mp = pd.read_csv(data / "masked_prices_survivorship.csv", parse_dates=["date"])
    mp = mp.set_index("date").sort_index()
    panels = raw_return_panels(mp)
    W = momentum_weights(mp)

    nets: dict[str, pd.Series] = {}
    for stage, rets in panels.items():
        gross = (W * rets).sum(axis=1)
        gross = gross.where(W.abs().sum(axis=1) > 0)
        to = turnover_from_weights(W, rets)
        nets[stage] = (gross - cost_bps / 1e4 * to).dropna().rename(stage)

    comp_rows = []
    for stage, s in nets.items():
        t = s[s.index >= TEST_START].dropna()
        ht, hp = hac_mean_tstat(t) if len(t) > 12 else (float("nan"), float("nan"))
        comp_rows.append(
            {
                "Panel": stage,
                "N_test": int(len(t)),
                "AnnRet_test": ann_return(t),
                "AnnVol_test": ann_vol(t),
                "Sharpe_test": sharpe(t),
                "MaxDD_test": max_drawdown(t),
                "HAC-t_test": ht,
                "HAC-p_test": hp,
            }
        )
    comp = pd.DataFrame(comp_rows).set_index("Panel")

    raw_vals = panels["raw"].stack().dropna()
    n_hit = int(((raw_vals.abs() > 0.40).sum()))
    diag = {
        "cap_hit_rate": float(n_hit / max(len(raw_vals), 1)),
        "n_capped_obs": float(n_hit),
        "worst_month_raw": float(nets["raw"][nets["raw"].index >= TEST_START].min()),
        "worst_month_final": float(
            nets["final"][nets["final"].index >= TEST_START].min()
        ),
    }
    monthly = pd.DataFrame(nets)
    monthly_test = monthly[monthly.index >= TEST_START].dropna()
    diag["max_abs_monthly_delta_raw_vs_final"] = float(
        (monthly_test["raw"] - monthly_test["final"]).abs().max()
    )
    diag["mean_monthly_delta_raw_vs_final"] = float(
        (monthly_test["raw"] - monthly_test["final"]).mean()
    )
    return comp, diag, monthly_test


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

    # Raw-return sensitivity: same weights, no cap / no winsorization.
    raw_comp, raw_diag, raw_monthly = raw_sensitivity(data, COST_BPS_BASE)
    raw_comp.to_csv(out / "raw_sensitivity_test.csv")

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

    # Overlay: stored (capped + winsorized) vs rebuilt raw net wealth.
    stored_test = net_a[net_a.index >= TEST_START].dropna()
    rw = pd.DataFrame(
        {
            "Stored (cap + winsorized)": (1 + stored_test).cumprod(),
            "Raw (no cap, no winsorize)": (
                1 + raw_monthly["raw"].loc[stored_test.index]
            ).cumprod(),
        }
    ).dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    rw.plot(ax=ax)
    ax.set_title("Test-window wealth: stored vs raw returns (10 bps)")
    ax.set_ylabel("Wealth")
    fig.tight_layout()
    fig.savefig(out / "raw_vs_stored_test.png", dpi=200)
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
        f.write(
            "raw sensitivity (test, 10bps): stored AnnRet={:.4f} Sharpe={:.3f}; "
            "raw AnnRet={:.4f} Sharpe={:.3f}\n".format(
                strat_tbl.loc["Test/OOS (2020-2024)", "AnnRet"],
                strat_tbl.loc["Test/OOS (2020-2024)", "Sharpe"],
                raw_comp.loc["raw", "AnnRet_test"],
                raw_comp.loc["raw", "Sharpe_test"],
            )
        )
        f.write(
            "cap hit rate={:.5f} ({:.0f} stock-months); "
            "max|raw-final| monthly={:.4f} mean delta={:.5f}\n".format(
                raw_diag["cap_hit_rate"],
                raw_diag["n_capped_obs"],
                raw_diag["max_abs_monthly_delta_raw_vs_final"],
                raw_diag["mean_monthly_delta_raw_vs_final"],
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
    print()
    print("Raw-return sensitivity (test window, 10 bps):")
    print(raw_comp.round(4).to_string())
    print(
        "cap hit rate={:.4%} worst_raw={:.2%} worst_final={:.2%} "
        "max|raw-final|={:.2%}".format(
            raw_diag["cap_hit_rate"],
            raw_diag["worst_month_raw"],
            raw_diag["worst_month_final"],
            raw_diag["max_abs_monthly_delta_raw_vs_final"],
        )
    )
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
