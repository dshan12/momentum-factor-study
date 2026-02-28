import pandas as pd
import numpy as np


def equal_weight_long_short(longs: pd.DataFrame, shorts: pd.DataFrame) -> pd.DataFrame:
    """
    Build equal-weight long-short weights.
    Long leg sums to +1, short leg sums to -1 each period.
    Total gross notional = 2, net = 0 (dollar neutral).
    No further normalization needed after this.
    """
    L = (longs > 0).astype(float)
    S = (shorts > 0).astype(float)
    lw = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = -S.div(S.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return lw.add(sw, fill_value=0.0)


def drift_weights(W_prev: pd.Series, rets_prev: pd.Series) -> pd.Series:
    """
    Mark weights to market using last period's returns, then re-normalize
    each leg independently back to ±1 to maintain dollar-neutrality.
    Used only for computing pre-trade weights in turnover calculation.
    """
    cols = W_prev.index.union(rets_prev.index)
    w = W_prev.reindex(cols).fillna(0.0)
    r = rets_prev.reindex(cols).fillna(0.0)
    w_drifted = w * (1.0 + r)

    long_mask = w_drifted > 0
    short_mask = w_drifted < 0
    long_sum = w_drifted[long_mask].sum()
    short_sum = w_drifted[short_mask].abs().sum()

    w_out = pd.Series(0.0, index=cols)
    if long_sum > 1e-12:
        w_out[long_mask] = w_drifted[long_mask] / long_sum
    if short_sum > 1e-12:
        w_out[short_mask] = w_drifted[short_mask] / short_sum  # keeps negative sign
    return w_out


def turnover_from_weights(W: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """
    One-way turnover = 0.5 * sum(|w_target - w_pre_rebalance|)
    where w_pre_rebalance is the drifted weight just before trading.
    """
    to = []
    for i in range(len(W)):
        if i == 0:
            to.append(0.0)
            continue
        w_pre = drift_weights(W.iloc[i - 1], rets.iloc[i - 1])
        w_t = W.iloc[i].reindex(W.columns).fillna(0.0)
        w_pre = w_pre.reindex(W.columns).fillna(0.0)
        to.append(0.5 * (w_t - w_pre).abs().sum())
    return pd.Series(to, index=W.index, name="turnover")


def apply_turnover_costs(
    gross: pd.Series, turnover: pd.Series, cost_bps: float
) -> pd.Series:
    net = gross - (cost_bps / 10_000.0) * turnover
    net.name = "strategy_net"
    return net
