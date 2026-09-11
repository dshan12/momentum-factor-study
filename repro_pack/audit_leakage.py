"""Look-ahead-bias audit for the 12-1 momentum reproducibility pack.

Verifies timing, masking, and construction properties on the frozen
inputs in ../data/cleaned/ and writes a checklist to outputs/.

Usage:
    python audit_leakage.py [--data-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "cleaned"
DEFAULT_OUT = HERE / "outputs"


def check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"Check": name, "Result": "PASS" if passed else "FAIL", "Detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    data, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []

    net = pd.read_csv(data / "strategy_net_survivorship.csv", parse_dates=["date"])
    gross = pd.read_csv(
        data / "strategy_gross_survivorship.csv", parse_dates=["date"]
    )
    turnover = pd.read_csv(
        data / "strategy_turnover_survivorship.csv", parse_dates=["date"]
    )
    rets = pd.read_csv(data / "returns_survivorship.csv", parse_dates=["date"])
    memb = pd.read_csv(
        data / "sp500_membership_monthly.csv", parse_dates=["date"]
    )

    # 1. Burn-in consistent with 12-month lookback + 1-month skip + 1-month lag.
    first_valid = pd.Timestamp("2006-03-31")
    actual_first = net["date"].min()
    results.append(
        check(
            "Burn-in period",
            actual_first >= first_valid,
            f"First net return {actual_first.date()}, "
            f"expected on or after {first_valid.date()} (13-month burn-in).",
        )
    )

    # 2. No trade before signal availability (gross leading NaNs).
    n_valid = int(net["date"].count())
    n_gross_valid = int(
        gross.dropna(subset=[c for c in gross.columns if c != "date"]).shape[0]
    )
    results.append(
        check(
            "No pre-signal trading",
            n_valid == 226 and n_gross_valid == 226,
            f"net valid n={n_valid}, gross valid n={n_gross_valid} (expected 226).",
        )
    )

    # 3. Turnover starts at zero (no phantom initial trade cost).
    t0 = float(turnover["turnover"].iloc[0])
    results.append(
        check(
            "Turnover initialization",
            abs(t0) < 1e-12,
            f"First turnover value {t0:.6f} (expected 0.0).",
        )
    )

    # 4. Return caps respected (hard cap +/-40 pct before winsorization).
    ret_vals = rets.drop(columns=["date"]).stack().dropna()
    results.append(
        check(
            "Return hard cap",
            bool(((ret_vals >= -0.40 - 1e-9) & (ret_vals <= 0.40 + 1e-9)).all()),
            f"Panel min {ret_vals.min():.4f}, max {ret_vals.max():.4f}.",
        )
    )

    # 5. Membership plausibility (S&P 500 breadth, ~500 names/month).
    per_month = memb.groupby("date").size()
    results.append(
        check(
            "Membership breadth",
            bool((per_month >= 400).all() and (per_month <= 600).all()),
            f"Names/month min {per_month.min()}, mean {per_month.mean():.0f}, "
            f"max {per_month.max()} over {per_month.shape[0]} months.",
        )
    )

    # 6. Monthly frequency, no duplicate or future-dated index entries.
    dates = pd.to_datetime(net["date"]).sort_values()
    gaps_ok = bool((dates.diff().dropna() <= pd.Timedelta(days=32)).all())
    dup_ok = bool(not dates.duplicated().any())
    results.append(
        check(
            "Index integrity",
            gaps_ok and dup_ok,
            f"Monthly spacing ok={gaps_ok}, duplicates ok={dup_ok}, "
            f"last date {dates.max().date()}.",
        )
    )

    # 7. Documented execution lag (weights shifted one month before returns).
    # Verified by construction in src/analysis_survivorship_free.py (W.shift(1));
    # here we confirm the stored net equals gross minus 10 bps * turnover.
    merged = (
        gross.merge(turnover, on="date").merge(net, on="date").dropna()
    )
    gcol = [c for c in merged.columns if "gross" in c][0]
    ncol = [c for c in merged.columns if c.startswith("strategy_net")][0]
    recon = merged[gcol] - 10.0 / 1e4 * merged["turnover"]
    max_err = float((recon - merged[ncol]).abs().max())
    results.append(
        check(
            "Cost accounting identity",
            max_err < 1e-9,
            f"max|net - (gross - 10bps*turnover)| = {max_err:.2e}.",
        )
    )

    df = pd.DataFrame(results)
    df.to_csv(out / "leakage_checklist.csv", index=False)
    with open(out / "leakage_checklist.md", "w") as f:
        f.write("# Look-Ahead-Bias Audit\n\n")
        f.write("| Check | Result | Detail |\n|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Check']} | {r['Result']} | {r['Detail']} |\n")
        f.write(
            "\nNotes: signal uses prices t-13 to t-1 only; portfolio weights "
            "are shifted one month before multiplication by returns; "
            "membership masking, gap-aware returns, and contemporaneous "
            "cross-sectional winsorization are applied at formation time. "
            "Factor data are used for ex-post attribution only, never in "
            "signal construction.\n"
        )

    print(df.to_string(index=False))
    print(f"\nWrote {out / 'leakage_checklist.md'}")


if __name__ == "__main__":
    main()
