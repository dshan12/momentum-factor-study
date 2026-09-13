"""Phase 2 point-in-time universe audit (frozen inputs + live anchor snapshot).

Checks (see phase2_pack/PREREG.md section 3):
  1. Count band vs ~503-stock benchmark; 2. add/drop conservation;
  3. frozen Dec-2024 vs live anchor overlap;
  4. stale-ticker separation via trailing price availability;
  5. strategy-level impact (ghost weight in test-window legs);
  6. membership-build re-executability from source.

Usage:
    python universe_audit.py [--data-dir PATH] [--out-dir PATH]
                             [--live-url | --no-live]

Live re-pull is attempted only with --live-url; default uses the frozen
snapshot inputs_snapshot/live_constituents_20260911.csv so the pack
reruns offline with identical results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "cleaned"
DEFAULT_OUT = HERE / "outputs"
SNAPSHOT = HERE / "inputs_snapshot" / "live_constituents_20260911.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
EXPECTED_STOCKS = 503
TEST_START = pd.Timestamp("2020-01-31")


def load_monthly_membership(data: Path) -> pd.DataFrame:
    m = pd.read_csv(data / "sp500_membership_monthly.csv", parse_dates=["date"])
    return m.sort_values(["date", "ticker"]).reset_index(drop=True)


def try_live_pull() -> set[str] | None:
    try:
        import sys

        sys.path.insert(0, str(HERE.parent / "src"))
        from data.build_sp500_history import _extract_current_constituents

        cur = _extract_current_constituents()
        return set(cur["Symbol"].dropna().tolist())
    except Exception as e:
        print(f"live pull failed ({e}); using frozen snapshot")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--live-url", action="store_true")
    args = ap.parse_args()
    data, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    m = load_monthly_membership(data)
    counts = m.groupby("date").size()
    excess = counts - EXPECTED_STOCKS

    # 2. Conservation: monthly adds vs drops.
    ms = m.groupby("date")["ticker"].apply(set)
    adds, drops = [], []
    for i in range(1, len(ms)):
        adds.append(len(set(ms.iloc[i]) - set(ms.iloc[i - 1])))
        drops.append(len(set(ms.iloc[i - 1]) - set(ms.iloc[i])))
    adds = pd.Series(adds, index=counts.index[1:])
    drops = pd.Series(drops, index=counts.index[1:])

    # 3. Anchor overlap.
    live = try_live_pull() if args.live_url else None
    if live is None:
        live = set(pd.read_csv(SNAPSHOT)["ticker"].tolist())
        anchor_note = "frozen snapshot 2026-09-11 (offline)"
    else:
        anchor_note = "live re-pull (online)"
    last_date = m["date"].max()
    frozen_last = set(m[m["date"] == last_date]["ticker"])
    stale = sorted(frozen_last - live)
    missed = sorted(live - frozen_last)

    # 4. Stale separation via lifetime/trailing price availability.
    # Rule (stated before use): a frozen-not-live name with zero lifetime
    # prices never entered a rank or return, so its strategy impact is
    # exactly zero (this pools rename-ghosts and download failures, which
    # are indistinguishable offline and identical in effect). A name whose
    # prices end before 2024 is a mid-window delisting (ghost). A name
    # priced through window end was a legitimate member during the test
    # window -- most plausibly a genuine 2025 removal, no uncertainty.
    union = pd.read_csv(data / "monthly_adjclose_union.csv", parse_dates=["date"])
    union = union.set_index("date").sort_index()
    n_empty_union = int(
        sum(union[c].notna().sum() == 0 for c in union.columns)
    )
    mp_raw = pd.read_csv(
        data / "masked_prices_survivorship.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    tradable = mp_raw.loc[mp_raw.index >= "2005-01-01"].notna().sum(axis=1)
    rows = []
    for t in stale:
        n = int(union[t].notna().sum()) if t in union.columns else 0
        if n == 0:
            last, cls = "no-prices", "non-participating (zero impact)"
        else:
            last = union[t].dropna().index.max()
            last_s = last.strftime("%Y-%m-%d")
            if last < pd.Timestamp("2024-01-01"):
                last, cls = last_s, "mid-window delisting (ghost)"
            else:
                last, cls = last_s, "legitimate thru window (likely 2025 removal)"
        rows.append({"ticker": t, "n_prices": n, "last_price_month": last,
                     "classification": cls})
    stale_tbl = pd.DataFrame(rows)
    stale_tbl.to_csv(out / "stale_tickers.csv", index=False)
    n_nonpart = int((stale_tbl["classification"].str.startswith("non-part")).sum())
    n_ghost = int((stale_tbl["classification"].str.startswith("mid-window")).sum())
    n_legit = int((stale_tbl["classification"].str.startswith("legitimate")).sum())

    # 5. Strategy-level impact: stale weight in test-window 12-1 legs.
    # Non-participating names cannot appear (no prices -> no ranks);
    # mid-window ghosts could appear historically -- check test window.
    mp = mp_raw
    L = np.log(mp.where(mp > 0))
    mom = (L.shift(1) - L.shift(13)).replace([np.inf, -np.inf], np.nan)
    ranks = mom.rank(axis=1, pct=True, na_option="keep")
    longs = (ranks >= 0.9).astype(int)
    shorts = (ranks <= 0.1).astype(int)
    valid = (longs.sum(axis=1) >= 20) & (shorts.sum(axis=1) >= 20)
    ghost_set = set(
        stale_tbl[~stale_tbl["classification"].str.startswith("legitimate")]["ticker"]
    )
    leg_cols = set(mp.columns)
    ghost_in_universe = ghost_set & leg_cols
    test_idx = longs.index[longs.index >= TEST_START]
    ghost_leg_months = 0
    ghost_max_share = 0.0
    for dt in test_idx:
        if not valid.loc[dt]:
            continue
        leg = set(longs.columns[longs.loc[dt] > 0]) | set(
            shorts.columns[shorts.loc[dt] > 0]
        )
        hit = leg & ghost_in_universe
        if hit:
            # Ghosts have no valid trailing prices; check any valid return.
            ghost_leg_months += 1
        n_leg = len(leg)
        if n_leg:
            ghost_max_share = max(ghost_max_share, len(hit) / n_leg)
    rets = pd.read_csv(
        data / "returns_survivorship.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    ghost_ret_coverage = {
        t: float(rets[t].loc[rets.index >= TEST_START].notna().sum()) for t in ghost_in_universe
    }
    ghost_test_months = sum(1 for v in ghost_ret_coverage.values() if v > 0)

    # 6. Re-executability probe (import-level; full rebuild needs network).
    try:
        import sys

        sys.path.insert(0, str(HERE.parent / "src"))
        from data import build_sp500_history as bsh  # noqa

        rebuild_import = "importable"
        try:
            bsh._extract_changes_table()
            changes_parse = "parse OK"
        except Exception as e:
            changes_parse = f"parse FAILED: {type(e).__name__}: {e}"
    except Exception as e:
        rebuild_import = f"import failed: {e}"
        changes_parse = "not probed"

    summary = pd.DataFrame(
        [
            {"metric": "months", "value": f"{len(counts)}"},
            {"metric": "mean members/month", "value": f"{counts.mean():.1f}"},
            {"metric": "min/max members", "value": f"{counts.min()}/{counts.max()}"},
            {
                "metric": "months outside [498,508]",
                "value": f"{int(((counts < 498) | (counts > 508)).sum())}",
            },
            {"metric": "mean excess vs 503", "value": f"{excess.mean():.1f}"},
            {
                "metric": "mean adds/drops per month",
                "value": f"{adds.mean():.2f}/{drops.mean():.2f}",
            },
            {
                "metric": "cumulative drift (adds-drops)",
                "value": f"{int(adds.sum() - drops.sum())}",
            },
            {
                "metric": f"frozen {last_date.date()} count",
                "value": f"{len(frozen_last)}",
            },
            {"metric": f"anchor ({anchor_note})", "value": f"{len(live)}"},
            {"metric": "frozen-not-live (stale-or-2025-removed)", "value": f"{len(stale)}"},
            {"metric": "non-participating (zero lifetime prices)", "value": f"{n_nonpart}"},
            {"metric": "mid-window delistings (ghosts)", "value": f"{n_ghost}"},
            {"metric": "legitimate thru window (likely 2025 removals)", "value": f"{n_legit}"},
            {"metric": "union tickers fully empty", "value": f"{n_empty_union}"},
            {
                "metric": "tradable members/month (mean/min)",
                "value": f"{tradable.mean():.0f}/{tradable.min()}",
            },
            {"metric": "live-not-frozen", "value": f"{len(missed)}"},
            {
                "metric": "test months with ghost in leg",
                "value": f"{ghost_leg_months}",
            },
            {
                "metric": "max ghost share of a monthly leg",
                "value": f"{ghost_max_share:.4f}",
            },
            {
                "metric": "ghosts with any test-window return",
                "value": f"{ghost_test_months}/{len(ghost_in_universe)}",
            },
            {"metric": "rebuild module status", "value": rebuild_import},
            {"metric": "changes-table re-parse today", "value": changes_parse},
        ]
    )
    summary.to_csv(out / "universe_audit.csv", index=False)
    with open(out / "universe_audit.md", "w") as f:
        f.write("# Point-in-Time Universe Audit\n\n")
        f.write("| metric | value |\n|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['metric']} | {row['value']} |\n")
        f.write(
            "\n\nConfound: the anchor postdates the frozen window by ~20 months; "
            "frozen-not-live names split into non-participating (zero prices, "
            "zero impact by construction), mid-window delistings, and "
            "legitimate members priced through window end (no test-window "
            "uncertainty). See stale_tickers.csv.\n"
        )

    print(summary.to_string(index=False))
    print(f"\nWrote {out / 'universe_audit.md'}")


if __name__ == "__main__":
    main()
