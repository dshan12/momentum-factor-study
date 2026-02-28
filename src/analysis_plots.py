import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data", "cleaned")
FIG_DIR = os.path.join(ROOT, "figures")

NET_PATH = os.path.join(DATA_DIR, "strategy_net_survivorship.csv")
GROSS_PATH = os.path.join(DATA_DIR, "strategy_gross_survivorship.csv")
FF_PATH = os.path.join(DATA_DIR, "ff5_umd_monthly.csv")


def load_series(path: str, name: str) -> pd.Series:
    s = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze("columns")
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = name
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    # Drop NaNs and sort — do NOT asfreq here, that creates phantom NaN rows
    return s.dropna().sort_index()


def cumulative_wealth(r: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Compound returns; NaN months are skipped (not treated as 0%)."""
    r_clean = r.dropna()
    return start_value * (1.0 + r_clean).cumprod()


def drawdown_series(r: pd.Series) -> pd.Series:
    w = cumulative_wealth(r)
    dd = (w / w.cummax()) - 1.0
    dd.name = r.name
    return dd


def rolling_sharpe(
    r: pd.Series, rf: pd.Series, window: int = 36, min_periods: int = 12
) -> pd.Series:
    df = pd.concat([r.dropna(), rf], axis=1, keys=["r", "rf"]).dropna(how="any")
    ex = df["r"] - df["rf"]
    mu = ex.rolling(window, min_periods=min_periods).mean()
    sd = ex.rolling(window, min_periods=min_periods).std()
    return (mu / (sd + 1e-12)) * np.sqrt(12.0)


def style_ax(ax, ylabel="", title="", zero_line=False):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load strategy series (already clean decimals, no NaN) ---
    net = load_series(NET_PATH, "Strategy (net)")
    gross = load_series(GROSS_PATH, "Strategy (gross)")

    # --- Load FF factors, trim to strategy sample period ---
    ff = pd.read_csv(FF_PATH, parse_dates=["date"]).set_index("date").sort_index()
    ff.index = ff.index + pd.offsets.MonthEnd(0)

    # Trim FF to match the strategy's actual date range
    start, end = net.index.min(), net.index.max()
    ff = ff.loc[start:end]

    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("US Market")
    umd = ff["UMD"].rename("UMD (Fama-French)")
    rf = ff["RF"]

    # Align all on the strategy's index (inner join)
    idx = net.index.intersection(mkt.index)
    net_a = net.loc[idx]
    gross_a = gross.reindex(idx).dropna()
    mkt_a = mkt.loc[idx]
    umd_a = umd.loc[idx]
    rf_a = rf.loc[idx]

    print(
        f"[plots] sample: {idx.min().date()} → {idx.max().date()}  ({len(idx)} months)"
    )
    print(f"[plots] net ann return:  {(1 + net_a.mean()) ** 12 - 1:.2%}")
    print(f"[plots] mkt ann return:  {(1 + mkt_a.mean()) ** 12 - 1:.2%}")
    print(f"[plots] UMD ann return:  {(1 + umd_a.mean()) ** 12 - 1:.2%}")

    # ---------------------------------------------------------------- #
    # Figure 1: Cumulative wealth                                       #
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 5))
    cumulative_wealth(net_a).plot(
        ax=ax, color="#e63946", lw=1.8, label="Strategy (net)"
    )
    cumulative_wealth(gross_a).plot(
        ax=ax, color="#e63946", lw=1.0, ls="--", alpha=0.55, label="Strategy (gross)"
    )
    cumulative_wealth(mkt_a).plot(ax=ax, color="#457b9d", lw=1.8, label="US Market")
    cumulative_wealth(umd_a).plot(
        ax=ax, color="#2a9d8f", lw=1.3, ls="-.", alpha=0.85, label="UMD (Fama-French)"
    )
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1fx"))
    ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.4)
    style_ax(
        ax,
        ylabel="Wealth (start = 1.0)",
        title="Cumulative Wealth — Momentum Strategy vs Benchmarks",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    f1 = os.path.join(FIG_DIR, "equity_curve.png")
    fig.savefig(f1, dpi=200)
    plt.close(fig)
    print(f"[✓] {f1}")

    # ---------------------------------------------------------------- #
    # Figure 2: Drawdown                                                #
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 4))
    dd_net = drawdown_series(net_a)
    dd_mkt = drawdown_series(mkt_a)
    dd_net.plot(ax=ax, color="#e63946", lw=1.4, label="Strategy (net)")
    dd_mkt.plot(ax=ax, color="#457b9d", lw=1.0, ls="--", alpha=0.7, label="US Market")
    ax.fill_between(dd_net.index, dd_net.values, 0, alpha=0.12, color="#e63946")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    style_ax(ax, ylabel="Drawdown", title="Drawdown — Strategy (net) vs US Market")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f2 = os.path.join(FIG_DIR, "drawdown.png")
    fig.savefig(f2, dpi=200)
    plt.close(fig)
    print(f"[✓] {f2}")

    # ---------------------------------------------------------------- #
    # Figure 3: Rolling 36-month Sharpe                                #
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 4))
    rolling_sharpe(net_a, rf_a).plot(
        ax=ax, color="#e63946", lw=1.4, label="Strategy (net)"
    )
    rolling_sharpe(mkt_a, rf_a).plot(
        ax=ax, color="#457b9d", lw=1.0, ls="--", alpha=0.7, label="US Market"
    )
    rolling_sharpe(umd_a, rf_a).plot(
        ax=ax, color="#2a9d8f", lw=1.0, ls="-.", alpha=0.7, label="UMD factor"
    )
    style_ax(
        ax,
        ylabel="Sharpe (36m rolling)",
        title="Rolling 36-Month Sharpe Ratio (excess over RF)",
        zero_line=True,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    f3 = os.path.join(FIG_DIR, "rolling_sharpe_36m.png")
    fig.savefig(f3, dpi=200)
    plt.close(fig)
    print(f"[✓] {f3}")

    # ---------------------------------------------------------------- #
    # Figure 4: Monthly return distribution                            #
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        net_a.values, bins=50, color="#e63946", alpha=0.7, edgecolor="white", lw=0.4
    )
    ax.axvline(
        net_a.mean(), color="black", lw=1.3, ls="--", label=f"Mean = {net_a.mean():.2%}"
    )
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.text(
        0.97,
        0.95,
        f"Skew: {net_a.skew():.2f}\nKurt: {net_a.kurtosis():.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    style_ax(ax, ylabel="Count", title="Monthly Return Distribution — Strategy (net)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f4 = os.path.join(FIG_DIR, "return_distribution.png")
    fig.savefig(f4, dpi=200)
    plt.close(fig)
    print(f"[✓] {f4}")


if __name__ == "__main__":
    main()
