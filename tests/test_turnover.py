import numpy as np
import pandas as pd
import pytest

from data.turnover import (
    apply_turnover_costs,
    drift_weights,
    equal_weight_long_short,
    turnover_from_weights,
)


class TestEqualWeightLongShort:
    def test_basic_long_short(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        tickers = ["A", "B", "C", "D"]
        longs = pd.DataFrame(
            {t: [1, 1, 0] for t in tickers[:2]}, index=dates
        ).reindex(columns=tickers, fill_value=0)
        shorts = pd.DataFrame(
            {t: [1, 1, 0] for t in tickers[2:]}, index=dates
        ).reindex(columns=tickers, fill_value=0)
        longs.iloc[2] = 0
        shorts.iloc[2] = 1

        W = equal_weight_long_short(longs, shorts)
        assert W.shape == (3, 4)
        assert pytest.approx(W.iloc[0].sum()) == 0.0
        assert pytest.approx(W.iloc[0]["A"]) == 0.5
        assert pytest.approx(W.iloc[0]["B"]) == 0.5
        assert pytest.approx(W.iloc[0]["C"]) == -0.5
        assert pytest.approx(W.iloc[0]["D"]) == -0.5

    def test_no_longs(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        tickers = ["A", "B"]
        longs = pd.DataFrame(0, index=dates, columns=tickers)
        shorts = pd.DataFrame(
            {t: [1, 0] for t in tickers}, index=dates
        )

        W = equal_weight_long_short(longs, shorts)
        assert (W == 0).all().all()

    def test_no_shorts(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        tickers = ["A", "B"]
        longs = pd.DataFrame(
            {t: [1, 0] for t in tickers}, index=dates
        )
        shorts = pd.DataFrame(0, index=dates, columns=tickers)

        W = equal_weight_long_short(longs, shorts)
        assert (W == 0).all().all()

    def test_all_nan_period(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        tickers = ["A", "B"]
        longs = pd.DataFrame(0, index=dates, columns=tickers)
        shorts = pd.DataFrame(0, index=dates, columns=tickers)
        longs.iloc[0] = 1
        shorts.iloc[1] = 1

        W = equal_weight_long_short(longs, shorts)
        assert W.iloc[0].sum() == pytest.approx(0.0)
        assert W.iloc[1].sum() == pytest.approx(0.0)
        assert W.iloc[1].isna().sum() == 0

    def test_long_sum_equals_1(self):
        dates = pd.date_range("2020-01-31", periods=1, freq="ME")
        tickers = ["A", "B", "C", "D", "E"]
        longs = pd.DataFrame({t: [1] for t in tickers[:3]}, index=dates)
        shorts = pd.DataFrame({t: [1] for t in tickers[3:]}, index=dates)

        W = equal_weight_long_short(longs, shorts)
        assert pytest.approx(W.iloc[0, :3].sum()) == 1.0
        assert pytest.approx(W.iloc[0, 3:].sum()) == -1.0


class TestDriftWeights:
    def test_no_drift(self):
        W_prev = pd.Series({"A": 0.5, "B": 0.5, "C": -1.0})
        rets_prev = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
        w = drift_weights(W_prev, rets_prev)
        assert pytest.approx(w["A"]) == 0.5
        assert pytest.approx(w["B"]) == 0.5
        assert pytest.approx(w["C"]) == -1.0

    def test_drift_positive_returns(self):
        W_prev = pd.Series({"A": 1.0, "B": 0.0})
        rets_prev = pd.Series({"A": 0.1, "B": 0.0})
        w = drift_weights(W_prev, rets_prev)
        assert pytest.approx(w["A"]) == 1.0
        assert w["A"] > 0
        assert w["B"] == 0.0

    def test_drift_long_leg_only(self):
        W_prev = pd.Series({"A": 0.5, "B": 0.5})
        rets_prev = pd.Series({"A": 0.1, "B": -0.1})
        w = drift_weights(W_prev, rets_prev)
        assert pytest.approx(w.sum()) == 1.0

    def test_drift_short_leg_only(self):
        W_prev = pd.Series({"A": -0.5, "B": -0.5})
        rets_prev = pd.Series({"A": 0.1, "B": -0.1})
        w = drift_weights(W_prev, rets_prev)
        assert pytest.approx(w.sum()) == -1.0

    def test_all_zero_weights(self):
        W_prev = pd.Series({"A": 0.0, "B": 0.0})
        rets_prev = pd.Series({"A": 0.1, "B": 0.0})
        w = drift_weights(W_prev, rets_prev)
        assert (w == 0).all()

    def test_non_overlapping_indices(self):
        W_prev = pd.Series({"A": 1.0})
        rets_prev = pd.Series({"B": 0.1})
        w = drift_weights(W_prev, rets_prev)
        assert pytest.approx(w["A"]) == 1.0
        assert w["B"] == 0.0


class TestTurnoverFromWeights:
    def test_zero_turnover_first_period(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        tickers = ["A", "B"]
        W = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates
        )
        rets = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates
        )
        to = turnover_from_weights(W, rets)
        assert to.iloc[0] == 0.0

    def test_no_churn(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        tickers = ["A", "B"]
        W = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates
        )
        rets = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates
        )
        to = turnover_from_weights(W, rets)
        for v in to.iloc[1:]:
            assert pytest.approx(v) == 0.0

    def test_complete_reversal(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        tickers = ["A", "B"]
        W = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=dates
        )
        rets = pd.DataFrame(
            {"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=dates
        )
        to = turnover_from_weights(W, rets)
        assert pytest.approx(to.iloc[1]) == 1.0

    def test_single_ticker(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        W = pd.DataFrame({"A": [1.0, 1.0]}, index=dates)
        rets = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
        to = turnover_from_weights(W, rets)
        assert to.iloc[0] == 0.0
        assert pytest.approx(to.iloc[1]) == 0.0

    def test_output_type_and_name(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        tickers = ["A", "B"]
        W = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates
        )
        rets = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates
        )
        to = turnover_from_weights(W, rets)
        assert isinstance(to, pd.Series)
        assert to.name == "turnover"


class TestApplyTurnoverCosts:
    def test_zero_cost(self):
        gross = pd.Series([0.01, 0.02, -0.01])
        to = pd.Series([0.1, 0.2, 0.15])
        net = apply_turnover_costs(gross, to, 0.0)
        assert net.name == "strategy_net"
        assert (net == gross).all()

    def test_positive_cost(self):
        gross = pd.Series([0.01, 0.02, -0.01])
        to = pd.Series([0.1, 0.2, 0.15])
        cost_bps = 10
        net = apply_turnover_costs(gross, to, cost_bps)
        expected = gross - (cost_bps / 10_000.0) * to
        assert pytest.approx(net.values) == expected.values

    def test_large_cost(self):
        gross = pd.Series([0.01])
        to = pd.Series([0.5])
        net = apply_turnover_costs(gross, to, 100)
        assert pytest.approx(net.iloc[0]) == 0.01 - (100 / 10_000.0) * 0.5

    def test_zero_turnover(self):
        gross = pd.Series([0.01, 0.02])
        to = pd.Series([0.0, 0.0])
        net = apply_turnover_costs(gross, to, 10)
        assert pytest.approx(net.values) == gross.values

    def test_empty_series(self):
        gross = pd.Series([], dtype=float)
        to = pd.Series([], dtype=float)
        net = apply_turnover_costs(gross, to, 10)
        assert len(net) == 0

    def test_output_name(self):
        gross = pd.Series([0.01])
        to = pd.Series([0.1])
        net = apply_turnover_costs(gross, to, 10)
        assert net.name == "strategy_net"
