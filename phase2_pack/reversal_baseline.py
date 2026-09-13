"""Phase 2 reversal baseline + decision summary (preregistered in PREREG.md).

Builds the 1-1 short-term reversal strategy (long prior-month losers,
short prior-month winners) with identical machinery to the 12-1 rule:
same universe, deciles, min-20 legs, equal weights, one-month execution
lag, drift-adjusted turnover, 10 bps costs. Ranking uses the same final
return panel, so the signal is the only difference.

Writes reversal_by_split.csv and decision_summary.md. Test window and
decision thresholds are unchanged from Phase 1.

Usage:
    python reversal_baseline.py [--data-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
COST_BPS = 10


def ann_return(r: pd.Series) -> float:
    return float((1.0 + r.dropna().mean()) ** 12 - 1.0)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(12.0))


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    rf_m = (1.0 + RF_ANNUAL) ** (1.0 / 12.0) - 1.0
    ex = r - rf_m
    return float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(12.0))


def max_drawdown(r: pd.Series) -> float:
    w = (1.0 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1.0).min())


def hac_tstat(r: pd.Series) -> tuple[float, float]:
    r = r.dropna()
    rf_m = (1.0 + RF_ANNUAL) ** (1.0 / 12.0) - 1.0
    y = (r - rf_m).values
    res = sm.OLS(y, np.ones((len(y), 1)), missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    return float(res.tvalues[0]), float(res.pvalues[0])


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    data, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rets = pd.read_csv(
        data / "returns_survivorship.csv", parse_dates=["date"]
    ).set_index("date").sort_index()

    # 1-1 signal: rank on the most recent monthly return (the month 12-1 skips).
    ranks = rets.rank(axis=1, pct=True, na_option="keep")
    longs = (ranks <= 0.1).astype(int)  # prior-month losers
    shorts = (ranks >= 0.9).astype(int)  # prior-month winners
    valid = (longs.sum(axis=1) >= 20) & (shorts.sum(axis=1) >= 20)
    longs = longs.where(valid, 0)
    shorts = shorts.where(valid, 0)
    lm = (longs > 0).astype(float)
    sm_ = (shorts > 0).astype(float)
    lw = lm.div(lm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = -sm_.div(sm_.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = lw.add(sw, fill_value=0.0).shift(1)

    gross = (W * rets).sum(axis=1)
    gross = gross.where(W.abs().sum(axis=1) > 0)
    to_list: list[float] = [0.0]
    for i in range(1, len(W)):
        w_pre = drift_weights(W.iloc[i - 1], rets.iloc[i - 1])
        w_t = W.iloc[i].reindex(W.columns).fillna(0.0)
        w_pre = w_pre.reindex(W.columns).fillna(0.0)
        to_list.append(0.5 * (w_t - w_pre).abs().sum())
    to = pd.Series(to_list, index=W.index, name="turnover")
    rev_net = (gross - COST_BPS / 1e4 * to).dropna().rename("reversal_net")
    rev_net.to_csv(out / "reversal_net.csv")

    mom_net = pd.read_csv(
        data / "strategy_net_survivorship.csv", parse_dates=["date"]
    ).set_index("date").sort_index()["strategy_net"]

    splits = {
        "IS (2006-2014)": (lambda s: s[s.index <= IS_END]),
        "Validation (2015-2019)": (
            lambda s: s[(s.index > IS_END) & (s.index <= VAL_END)]
        ),
        "Test/OOS (2020-2024)": (lambda s: s[s.index >= TEST_START]),
        "Full (2006-2024)": (lambda s: s),
    }
    rows = []
    for label, fn in splits.items():
        r = fn(rev_net).dropna()
        t, p = hac_tstat(r)
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
    rev_tbl = pd.DataFrame(rows).set_index("Split")
    rev_tbl.to_csv(out / "reversal_by_split.csv")

    mt = mom_net[mom_net.index >= TEST_START].dropna()
    rt = rev_net[rev_net.index >= TEST_START].dropna()
    t_m, p_m = hac_tstat(mt)
    t_r, p_r = hac_tstat(rt)
    lines = [
        "# Phase 2 Decision Summary (locked test 2020-2024)",
        "",
        "## Primary claim: sign of 12-1 net premium",
        f"- mean monthly = {mt.mean():.5f} ({'POSITIVE' if mt.mean() > 0 else 'NEGATIVE'})",
        f"- Sharpe = {sharpe(mt):.3f} ({'PASS' if sharpe(mt) > 0 else 'FAIL'})",
        f"- HAC(6) p = {p_m:.4f} ({'PASS' if p_m < 0.05 else 'FAIL'})",
        "- Verdict: REJECTED (unchanged from Phase 1)."
        if not (mt.mean() > 0 and sharpe(mt) > 0 and p_m < 0.05)
        else "- Verdict: ACCEPTED.",
        "",
        "## Secondary evidence: FF5+UMD alpha (interpretive only)",
        "- See Phase 1 ff5_umd_alpha_test.csv (alpha -4.93%, p 0.12).",
        "- Cannot overturn the primary verdict in either direction.",
        "",
        "## Context: preregistered 1-1 reversal baseline (not decisive)",
        f"- Reversal test AnnRet = {ann_return(rt):.4f}, Sharpe = {sharpe(rt):.3f}, "
        f"MaxDD = {max_drawdown(rt):.4f}, HAC-t = {t_r:.2f}, p = {p_r:.4f}, N = {len(rt)}.",
        f"- Momentum test AnnRet = {ann_return(mt):.4f}, Sharpe = {sharpe(mt):.3f}.",
        "",
        "## Universe audit impact",
        "- See universe_audit.md. Frozen test-window inputs are unchanged; "
        "the audit bounds uncertainty only.",
    ]
    with open(out / "decision_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(rev_tbl.round(4).to_string())
    print(f"\nMomentum test AnnRet={ann_return(mt):.4f} Sharpe={sharpe(mt):.3f}")
    print(f"Wrote {out / 'decision_summary.md'}")


if __name__ == "__main__":
    main()
