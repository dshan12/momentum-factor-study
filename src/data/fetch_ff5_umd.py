from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
OUT_PATH: str = os.path.join(ROOT, "data", "cleaned", "ff5_umd_monthly.csv")

FF5_URL: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
UMD_URL: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

YYYYMM_RE: re.Pattern[str] = re.compile(r"^\s*(\d{6})\b")


def _read_csv_block_from_zip(
    url: str, expected_header_hints: List[str]
) -> pd.DataFrame:
    """Download a zip file from Ken French's data library and parse the monthly CSV block.

    The zip contains a CSV file with a descriptive header, followed by the monthly
    data table.  This function locates the data by scanning for the first YYYYMM row.

    Args:
        url: URL of the zip file.
        expected_header_hints: Column-name substrings used to locate the header row.

    Returns:
        DataFrame with a 'date' column (month-end) and factor columns.

    Raises:
        RuntimeError: If the monthly table start cannot be found.
    """
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = next(
        (n for n in z.namelist() if n.lower().endswith(".csv")), z.namelist()[0]
    )
    with z.open(name) as f:
        text = f.read().decode("latin-1")

    lines = text.splitlines()

    try:
        start = next(i for i, ln in enumerate(lines) if YYYYMM_RE.match(ln))
    except StopIteration as e:
        raise RuntimeError("Could not find start of monthly table (YYYYMM).") from e

    header_idx: Optional[int] = None
    scan_above = range(max(0, start - 10), start)[::-1]
    for i in scan_above:
        ln_low = lines[i].lower()
        if any(hint.lower() in ln_low for hint in expected_header_hints):
            header_idx = i
            break
    if header_idx is None:
        header_idx = start - 1

    header_line = lines[header_idx]
    headers = [h.strip() for h in header_line.split(",")]

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            end = j
            break
        if lines[j].lower().startswith(("annual", "annually", "yearly")):
            end = j
            break

    block = "\n".join([",".join(headers)] + lines[start:end])

    df = pd.read_csv(io.StringIO(block))
    first = df.columns[0]
    df = df.rename(columns={first: "yyyymm"})
    df["yyyymm"] = df["yyyymm"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df[df["yyyymm"].notna()].copy()
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.drop(columns=["yyyymm"])
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    """Download FF5 and momentum factors, merge, and save to CSV."""
    ff5 = _read_csv_block_from_zip(
        FF5_URL, expected_header_hints=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    )
    keep_ff5 = [
        c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] if c in ff5.columns
    ]
    if not keep_ff5:
        raise RuntimeError(
            f"FF5 expected columns not found; got: {ff5.columns.tolist()}"
        )
    ff5 = ff5[["date"] + keep_ff5]

    umd = _read_csv_block_from_zip(
        UMD_URL,
        expected_header_hints=["UMD", "Mom"],
    )
    mom_candidates = [
        c for c in umd.columns if c.strip().lower().startswith(("umd", "mom"))
    ]
    if not mom_candidates:
        raise RuntimeError(f"Momentum column not found; got: {umd.columns.tolist()}")
    umd = umd.rename(columns={mom_candidates[0]: "UMD"})[["date", "UMD"]]

    f = ff5.merge(umd, on="date", how="inner").sort_values("date")
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]:
        if c in f.columns:
            f[c] = f[c] / 100.0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    f.to_csv(OUT_PATH, index=False)
    logger.info("Saved %s shape=%s", OUT_PATH, f.shape)


if __name__ == "__main__":
    main()
