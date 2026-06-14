from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cleaned"
PAPER_TABLES = ROOT / "paper" / "tables"

NET_PATH = DATA / "strategy_net_survivorship.csv"
FF_PATH = DATA / "ff5_umd_monthly.csv"

OUT_CSV = DATA / "summary_stats.csv"
OUT_TEX = PAPER_TABLES / "summary_stats.tex"


def load_series(path: Path, name: str) -> pd.Series:
    """Load a single-column CSV as a monthly-end Series.

    Args:
        path: Path to the CSV file.
        name: Name to assign to the returned Series.

    Returns:
        A sorted monthly-end-frequency Series.
    """
    s = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze("columns")
    if isinstance(s, pd.DataFrame):
        if s.shape[1] != 1:
            raise ValueError(f"Expected 1 column in {path}, got {s.shape[1]}")
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index().asfreq("ME")


def max_drawdown(r: pd.Series) -> float:
    """Compute the maximum drawdown from a series of returns.

    Args:
        r: Series of monthly returns (decimals).

    Returns:
        Maximum drawdown as a negative decimal.
    """
    w = (1.0 + r.fillna(0.0)).cumprod()
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return float(dd.min())


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.02) -> float:
    """Compute annualized Sharpe ratio from monthly return series.

    Args:
        r: Series of monthly returns (decimals).
        rf_annual: Annual risk-free rate (default 0.02 for 2%).

    Returns:
        Annualized Sharpe ratio.
    """
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    ex = r - rf_m
    return (ex.mean() / (ex.std() + 1e-12)) * np.sqrt(12.0)


def ann_return(r: pd.Series) -> float:
    """Annualize a mean monthly return.

    Args:
        r: Series of monthly returns (decimals).

    Returns:
        Annualized return as a decimal.
    """
    return (1.0 + r.mean()) ** 12 - 1.0


def build_benchmark_from_factors(ff: pd.DataFrame) -> pd.Series:
    """Build a US market total return proxy from FF factor data.

    The market return is computed as Mkt-RF + RF (decimal monthly).

    Args:
        ff: DataFrame with columns 'Mkt-RF' and 'RF'.

    Returns:
        Monthly-end Series of US market total returns.
    """
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("us_market")
    return mkt.asfreq("ME")


def summary_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a summary statistics table from a dict of return series.

    Args:
        series: Dictionary mapping label to monthly return Series.

    Returns:
        DataFrame indexed by label with columns for count, mean, vol,
        Sharpe, ann return, ann vol, skew, kurtosis, and max drawdown.
    """
    rows = []
    for label, r in series.items():
        r = r.dropna()
        rows.append(
            {
                "Series": label,
                "N": int(r.shape[0]),
                "Mean (m)": r.mean(),
                "Vol (m)": r.std(),
                "Sharpe (2% rf)": sharpe_ratio(r),
                "Ann Return": ann_return(r),
                "Ann Vol": r.std() * np.sqrt(12.0),
                "Skew": r.skew(),
                "Kurtosis": r.kurtosis(),
                "Max DD": max_drawdown(r),
            }
        )
    df = pd.DataFrame(rows).set_index("Series")
    return df


def save_latex(table: pd.DataFrame, path: Path) -> None:
    """Save a summary statistics table as a LaTeX file.

    Args:
        table: Summary statistics DataFrame.
        path: Output path for the .tex file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {
        "N": "{:d}".format,
        "Mean (m)": "{:.3%}".format,
        "Vol (m)": "{:.3%}".format,
        "Sharpe (2% rf)": "{:.2f}".format,
        "Ann Return": "{:.2%}".format,
        "Ann Vol": "{:.2%}".format,
        "Skew": "{:.2f}".format,
        "Kurtosis": "{:.2f}".format,
        "Max DD": "{:.2%}".format,
    }
    disp = table.copy()
    for c, f in fmt.items():
        disp[c] = [f(x) for x in disp[c]]

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Summary statistics (monthly, decimal returns)}",
        r"\label{tab:summary_stats}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Series & $N$ & Mean (m) & Vol (m) & Sharpe & Ann Ret & Ann Vol & Skew & Kurtosis & Max DD \\",
        r"\midrule",
    ]
    for idx, row in disp.iterrows():
        lines.append(
            f"{idx} & {row['N']} & {row['Mean (m)']} & {row['Vol (m)']} & "
            f"{row['Sharpe (2% rf)']} & {row['Ann Return']} & {row['Ann Vol']} & "
            f"{row['Skew']} & {row['Kurtosis']} & {row['Max DD']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Load strategy and benchmark returns, build summary table, and save."""
    net = load_series(NET_PATH, "Strategy (net)")
    ff = (
        pd.read_csv(FF_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("ME")
    )
    bench = build_benchmark_from_factors(ff).rename("US Market")

    idx = net.index.intersection(bench.index)
    net, bench = net.loc[idx], bench.loc[idx]

    table = summary_table({"Strategy (net)": net, "US Market": bench})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV)
    save_latex(table, OUT_TEX)

    disp = table.copy()
    logger.info("=== Summary statistics (monthly series) ===")
    with pd.option_context("display.float_format", "{:.6f}".format):
        logger.info("\n%s", disp)

    logger.info("Saved CSV: %s", OUT_CSV)
    logger.info("Saved LaTeX: %s", OUT_TEX)


if __name__ == "__main__":
    main()
