from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cleaned"

FACTORS_CSV = DATA / "ff5_umd_monthly.csv"
GROSS_CSV = DATA / "strategy_gross_survivorship.csv"
NET_CSV = DATA / "strategy_net_survivorship.csv"

OUT_CSV = DATA / "ff_regression_table.csv"
OUT_TEX = DATA / "ff_regression_table.tex"

MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+UMD": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"],
}
HAC_LAGS = 6


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
            raise ValueError(f"Expected single-column series in {path}")
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index().asfreq("ME")


def load_factors(path: Path) -> pd.DataFrame:
    """Load Fama-French factors from CSV.

    Args:
        path: Path to the FF CSV file.

    Returns:
        DataFrame with factor columns indexed by date.

    Raises:
        ValueError: If required factor columns are missing.
    """
    f = (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("ME")
    )
    need = {"RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"}
    missing = need - set(f.columns)
    if missing:
        raise ValueError(f"Missing factor columns in {path}: {missing}")
    return f


def regress_excess(
    ret: pd.Series, fac: pd.DataFrame, cols: list[str], lags: int = HAC_LAGS
) -> RegressionResultsWrapper:
    """Regress excess returns on a set of factor portfolios.

    Runs OLS with Newey-West (HAC) standard errors.

    Args:
        ret: Monthly return series (decimal).
        fac: Factor DataFrame including 'RF' column.
        cols: List of factor column names to include.
        lags: Number of HAC lags (default HAC_LAGS).

    Returns:
        Fitted OLS regression results.
    """
    y = (ret - fac["RF"]).dropna()
    X = fac.loc[y.index, cols].dropna()
    y = y.loc[X.index]
    X = sm.add_constant(X)
    res = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


def to_ann(m: float) -> float:
    """Annualize a monthly return.

    Args:
        m: Monthly return as a decimal.

    Returns:
        Annualized return as a decimal.
    """
    return (1.0 + m) ** 12 - 1.0


def stars(p: float) -> str:
    """Return significance stars for a p-value.

    Args:
        p: p-value from a statistical test.

    Returns:
        '***' if p < 0.01, '**' if p < 0.05, '*' if p < 0.10, else ''.
    """
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main() -> None:
    """Run factor regressions for gross and net strategy returns."""
    fac = load_factors(FACTORS_CSV)
    gross = load_series(GROSS_CSV, "gross")
    net = load_series(NET_CSV, "net")

    idx = gross.index.intersection(net.index).intersection(fac.index)
    gross, net, fac = gross.loc[idx].dropna(), net.loc[idx].dropna(), fac.loc[idx]

    rows = []
    logger.info("=== Factor regressions with Newey–West (HAC) SEs ===")
    for label, series in [("GROSS", gross), ("NET", net)]:
        logger.info("--- %s ---", label)
        for model_name, cols in MODELS.items():
            res = regress_excess(series, fac, cols, lags=HAC_LAGS)
            alpha_m = float(res.params["const"])
            alpha_t = float(res.tvalues["const"])
            alpha_p = float(res.pvalues["const"])
            alpha_ann = to_ann(alpha_m)
            row = {
                "Series": label,
                "Model": model_name,
                "Alpha_m": alpha_m,
                "Alpha_ann": alpha_ann,
                "Alpha_t": alpha_t,
                "Alpha_p": alpha_p,
                "R2": float(res.rsquared),
                "N": int(res.nobs),
            }
            for c in cols:
                row[f"beta_{c}"] = float(res.params.get(c, np.nan))
                row[f"t_{c}"] = float(res.tvalues.get(c, np.nan))
            rows.append(row)

            logger.info(
                "%7s  \u03b1(ann)=% .2f%%  t=% .2f%s  R\u00b2=% .3f  n=%d",
                model_name,
                alpha_ann * 100,
                alpha_t,
                stars(alpha_p),
                res.rsquared,
                int(res.nobs),
            )

    out = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Saved regression table: %s", OUT_CSV)

    def fmt_pct(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.2%}"

    def fmt_num(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.2f}"

    panel = out[out["Series"] == "NET"].copy()
    panel["Alpha"] = panel.apply(
        lambda r: f"{fmt_pct(r['Alpha_ann'])} ({fmt_num(r['Alpha_t'])})", axis=1
    )
    panel = (
        panel[["Model", "Alpha", "R2", "N"]].set_index("Model").reindex(MODELS.keys())
    )

    latex = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Factor Regressions (NET returns; HAC %d lags)}" % HAC_LAGS,
        r"\label{tab:ff_regressions}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & $\alpha$ (annual) [t] & $R^2$ & $n$ \\",
        r"\midrule",
    ]
    for m in MODELS.keys():
        rA = panel.loc[m, "Alpha"]
        rR2 = fmt_num(panel.loc[m, "R2"])
        rN = int(panel.loc[m, "N"])
        latex.append(f"{m} & {rA} & {rR2} & {rN} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(latex))
    logger.info("Saved LaTeX table: %s", OUT_TEX)


if __name__ == "__main__":
    main()
