import re
import os
import typing as T
import datetime as dt
from dataclasses import dataclass
from collections import defaultdict
from io import StringIO

import pandas as pd
import requests

WIKI_CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

YF_TICKER_FIX = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def normalize_ticker(t: str) -> str:
    if pd.isna(t):
        return None
    t = str(t).strip().upper().replace(" ", "")
    if t in YF_TICKER_FIX:
        return YF_TICKER_FIX[t]
    t = re.sub(r"\.(?=[A-Z0-9]+$)", "-", t)
    return t if t else None


def parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    try:
        return pd.to_datetime(s, errors="raise").normalize()
    except Exception:
        try:
            d = pd.to_datetime(s, errors="raise")
            return (d + pd.offsets.MonthEnd(0)).normalize()
        except Exception:
            return pd.NaT


@dataclass
class ChangeEvent:
    date: pd.Timestamp
    added: T.List[str]
    removed: T.List[str]
    raw: dict


def _extract_current_constituents() -> pd.DataFrame:
    html = fetch_html(WIKI_CURRENT_URL)
    df_list = pd.read_html(StringIO(html), flavor="bs4")
    candidates = [
        df
        for df in df_list
        if any(str(col).strip().lower() in ["symbol", "ticker"] for col in df.columns)
    ]
    if not candidates:
        raise RuntimeError("Failed to extract current constituents from Wikipedia.")
    cur = candidates[0].copy()

    # Rename columns to standard names
    colmap = {}
    for c in cur.columns:
        lc = str(c).strip().lower()
        if lc in ["symbol", "ticker", "code"]:
            colmap[c] = "Symbol"
        elif lc in ["security", "company", "name"]:
            colmap[c] = "Security"
        elif "gics" in lc and "sector" in lc:
            colmap[c] = "GICS Sector"
    cur = cur.rename(columns=colmap)
    cur["Symbol"] = cur["Symbol"].apply(normalize_ticker)
    cur = cur.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    return cur


def _extract_changes_table() -> pd.DataFrame:
    """
    Parse the Wikipedia S&P 500 changes table.
    The table has multi-level headers like:
        Date | Added (Ticker / Security) | Removed (Ticker / Security) | Reason
    We flatten and extract ticker columns directly — they are plain ticker strings,
    NOT wrapped in parentheses. The old regex was wrong and dropped most events.
    """
    html = fetch_html(WIKI_CURRENT_URL)
    all_tables = pd.read_html(StringIO(html), flavor="bs4", header=[0, 1])

    changes_df = None
    for t in all_tables:
        # Flatten MultiIndex columns
        if isinstance(t.columns, pd.MultiIndex):
            flat = [
                " ".join(str(x).strip() for x in col if "Unnamed" not in str(x)).strip()
                for col in t.columns
            ]
        else:
            flat = [str(c).strip() for c in t.columns]

        col_lower = [c.lower() for c in flat]
        has_date = any("date" in c for c in col_lower)
        has_added = any("added" in c for c in col_lower)
        has_removed = any("removed" in c for c in col_lower)

        if has_date and has_added and has_removed:
            t.columns = flat
            changes_df = t
            break

    if changes_df is None:
        raise RuntimeError(
            "Could not find S&P 500 changes table on Wikipedia. "
            "Column names may have changed — inspect the page manually."
        )

    # Identify the relevant columns by fuzzy name matching
    col_map = {}
    for c in changes_df.columns:
        lc = c.lower()
        if "date" in lc and "date" not in col_map:
            col_map["Date"] = c
        elif "added" in lc and "ticker" in lc and "Added_Ticker" not in col_map:
            col_map["Added_Ticker"] = c
        elif "added" in lc and "security" in lc and "Added_Security" not in col_map:
            col_map["Added_Security"] = c
        elif "removed" in lc and "ticker" in lc and "Removed_Ticker" not in col_map:
            col_map["Removed_Ticker"] = c
        elif "removed" in lc and "security" in lc and "Removed_Security" not in col_map:
            col_map["Removed_Security"] = c
        elif "reason" in lc and "Reason" not in col_map:
            col_map["Reason"] = c

    required = ["Date", "Added_Ticker", "Removed_Ticker"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise RuntimeError(
            f"Changes table missing expected columns: {missing}. "
            f"Found columns: {list(changes_df.columns)}"
        )

    result = pd.DataFrame()
    result["Date"] = changes_df[col_map["Date"]].apply(parse_date)
    # FIX: tickers are plain strings in these columns, no parentheses needed
    result["Added"] = changes_df[col_map["Added_Ticker"]].apply(
        lambda x: normalize_ticker(str(x).split("\n")[0].strip())
        if pd.notna(x)
        else None
    )
    result["Removed"] = changes_df[col_map["Removed_Ticker"]].apply(
        lambda x: normalize_ticker(str(x).split("\n")[0].strip())
        if pd.notna(x)
        else None
    )
    if "Reason" in col_map:
        result["Reason"] = changes_df[col_map["Reason"]]

    result = result.dropna(subset=["Date"])
    result = result[result["Date"] >= pd.Timestamp("1990-01-01")]
    return result.sort_values("Date").reset_index(drop=True)


def _make_events(changes: pd.DataFrame) -> T.List[ChangeEvent]:
    events = []
    for _, row in changes.iterrows():
        added = [row["Added"]] if pd.notna(row.get("Added")) and row["Added"] else []
        removed = (
            [row["Removed"]] if pd.notna(row.get("Removed")) and row["Removed"] else []
        )
        if not added and not removed:
            continue
        events.append(
            ChangeEvent(
                date=row["Date"],
                added=added,
                removed=removed,
                raw=row.to_dict(),
            )
        )
    return events


def build_membership_timeline(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a dataframe of daily membership from 'start' to 'end'.
    Strategy:
        1. Get current set from Wikipedia current-constituents table.
        2. Get all historical changes from the changes table.
        3. Roll back from today to 'start' by reversing each change.
        4. Roll forward day-by-day applying changes.
    """
    current = _extract_current_constituents()
    current_set = set(current["Symbol"].dropna().tolist())

    changes = _extract_changes_table()
    events = _make_events(changes)
    events.sort(key=lambda e: e.date)

    print(f"[*] Parsed {len(events)} change events from Wikipedia.")

    # Roll back from current to reconstruct membership at 'start'
    backward = [e for e in events if e.date > start]
    backward_rev = sorted(backward, key=lambda e: e.date, reverse=True)
    memb = set(current_set)
    for e in backward_rev:
        for a in e.added:
            memb.discard(a)
        for r in e.removed:
            if r:
                memb.add(r)

    print(f"[*] Reconstructed {len(memb)} members at start date {start.date()}.")

    # Roll forward
    by_date: T.DefaultDict[pd.Timestamp, T.List[ChangeEvent]] = defaultdict(list)
    for e in events:
        if start <= e.date <= end:
            by_date[e.date.normalize()].append(e)

    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    daily_rows = []
    current_members = set(memb)

    for d in days:
        if d in by_date:
            for ev in by_date[d]:
                for r in ev.removed:
                    current_members.discard(r)
                for a in ev.added:
                    if a:
                        current_members.add(a)
        for t in current_members:
            daily_rows.append((d, t, 1))

    daily = pd.DataFrame(daily_rows, columns=["date", "ticker", "in_index"])
    return daily


def monthly_panel_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily["month"] = pd.to_datetime(daily["date"]) + pd.offsets.MonthEnd(0)
    monthly = (
        daily.groupby(["month", "ticker"])["in_index"]
        .max()
        .reset_index()
        .rename(columns={"month": "date"})
    )
    return monthly


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", default="data/cleaned/sp500_membership_monthly.csv")
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    print("[*] Building survivorship-bias–free membership…")
    daily = build_membership_timeline(start, end)
    monthly = (
        monthly_panel_from_daily(daily)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    monthly["ticker"] = monthly["ticker"].apply(normalize_ticker)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    monthly.to_csv(args.out, index=False)
    print(f"[✓] Wrote {args.out} with {len(monthly):,} rows.")

    # Quick sanity check
    counts = monthly.groupby("date")["ticker"].count()
    print(
        f"[*] Avg members/month: {counts.mean():.0f} | Min: {counts.min()} | Max: {counts.max()}"
    )


if __name__ == "__main__":
    main()
