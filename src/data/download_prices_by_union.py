from __future__ import annotations

import logging
import os
import time
from typing import Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_ticker(t: str) -> Optional[str]:
    """Normalize a ticker symbol by uppercasing and replacing dots with hyphens.

    Args:
        t: Raw ticker string.

    Returns:
        Normalized ticker string, or None if input is NA.
    """
    if pd.isna(t):
        return None
    t = str(t).upper().strip().replace(" ", "")
    t = t.replace(".", "-")
    return t


def read_membership(path: str) -> List[str]:
    """Read ticker symbols from a membership CSV file.

    Args:
        path: Path to the monthly membership CSV with columns date, ticker, in_index.

    Returns:
        Sorted list of unique normalized ticker symbols.
    """
    m = pd.read_csv(path, parse_dates=["date"])
    tickers = sorted(set(m["ticker"].dropna().map(normalize_ticker)))
    return [t for t in tickers if t]


def batch(iterable: List[str], n: int = 25) -> Generator[List[str], None, None]:
    """Yield successive n-sized chunks from iterable.

    Args:
        iterable: The list to split into batches.
        n: Batch size (default 25).

    Yields:
        Non-overlapping sub-lists of size n (last may be shorter).
    """
    length = len(iterable)
    for i in range(0, length, n):
        yield iterable[i : i + n]


def download_batch(
    tickers: List[str], start: str, end: str, sleep_s: float = 0.3
) -> pd.DataFrame:
    """Download monthly adjusted close prices for a batch of tickers via yfinance.

    Args:
        tickers: List of ticker symbols to download.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        sleep_s: Seconds to sleep after download to rate-limit (default 0.3).

    Returns:
        DataFrame with columns date and one column per ticker.
    """
    import yfinance as yf

    df = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        out: List[pd.Series] = []
        for t in tickers:
            if t in df.columns.get_level_values(0):
                s = df[t]["Close"].rename(t)
                out.append(s)
        if not out:
            return pd.DataFrame()
        wide = pd.concat(out, axis=1)
    else:
        wide = df["Close"].to_frame(tickers[0]) if "Close" in df else pd.DataFrame()
    if wide.empty:
        return wide

    wide = wide.reset_index().rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"]) + pd.offsets.MonthEnd(0)
    time.sleep(sleep_s)
    return wide


def robust_download(all_tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download prices for all tickers with retry logic on failure.

    Args:
        all_tickers: Full list of ticker symbols to download.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with date column and one price column per ticker.
    """
    frames: List[pd.DataFrame] = []
    for grp in batch(all_tickers, n=25):
        tries = 0
        last_err: Optional[Exception] = None
        while tries < 3:
            try:
                part = download_batch(grp, start, end)
                if not part.empty:
                    frames.append(part)
                break
            except Exception as e:
                last_err = e
                tries += 1
                time.sleep(1 * tries)
        if tries == 3 and last_err:
            logger.warning("Failed batch %s... len=%d: %s", grp[:3], len(grp), last_err)
    if not frames:
        return pd.DataFrame(columns=["date"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date").drop_duplicates(subset=["date"])
    return out


def main() -> None:
    """Parse arguments and download monthly adjusted close prices for S&P 500 members."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--membership", default="data/cleaned/sp500_membership_monthly.csv"
    )
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--out", default="data/cleaned/monthly_adjclose_union.csv")
    args = parser.parse_args()

    tickers = read_membership(args.membership)
    if not tickers:
        raise RuntimeError("No tickers found in membership file.")

    prices = robust_download(tickers, args.start, args.end)
    if prices.empty:
        raise RuntimeError("No prices downloaded.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prices.to_csv(args.out, index=False)
    logger.info("Wrote %s shape=%s", args.out, prices.shape)


if __name__ == "__main__":
    main()
