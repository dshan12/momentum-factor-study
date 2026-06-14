import numpy as np
import pandas as pd
import pytest

from analysis_summary import (
    ann_return,
    build_benchmark_from_factors,
    load_series,
    max_drawdown,
    sharpe_ratio,
    summary_table,
)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        r = pd.Series([0.01, 0.01, 0.01])
        assert pytest.approx(max_drawdown(r)) == 0.0

    def test_simple_drawdown(self):
        r = pd.Series([0.0, -0.1, 0.0])
        dd = max_drawdown(r)
        assert dd < 0
        assert pytest.approx(dd, abs=1e-6) == -0.1

    def test_drawdown_from_peak(self):
        r = pd.Series([0.10, -0.05, -0.05])
        dd = max_drawdown(r)
        expect = (1.10 * 0.95 * 0.95) / 1.10 - 1.0
        assert pytest.approx(dd, abs=1e-6) == expect

    def test_continuous_decline(self):
        r = pd.Series([-0.01, -0.02, -0.03])
        dd = max_drawdown(r)
        w = (1 + r).cumprod()
        expected = float((w / w.cummax() - 1).min())
        assert pytest.approx(dd) == expected

    def test_single_element(self):
        r = pd.Series([0.05])
        assert pytest.approx(max_drawdown(r)) == 0.0

    def test_contains_nan(self):
        r = pd.Series([0.01, np.nan, -0.05, 0.02])
        dd = max_drawdown(r)
        assert dd <= 0
        assert dd > -0.06


class TestSharpeRatio:
    def test_zero_risk_free(self):
        r = pd.Series(np.full(12, 0.01))
        sr = sharpe_ratio(r, rf_annual=0.0)
        assert pytest.approx(sr, abs=1e-4) == 0.0

    def test_positive_excess(self):
        r = pd.Series(np.full(120, 0.01))
        sr = sharpe_ratio(r, rf_annual=0.0)
        assert sr > 0

    def test_with_risk_free(self):
        r = pd.Series(np.full(12, 0.01))
        sr = sharpe_ratio(r, rf_annual=0.12)
        assert sr < 0

    def test_high_sharpe(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.02, 0.02, 120))
        sr = sharpe_ratio(r, rf_annual=0.0)
        assert sr > 1.0

    def test_constant_series(self):
        r = pd.Series(np.full(60, 0.005))
        sr = sharpe_ratio(r, rf_annual=0.0)
        assert np.isinf(sr) or sr > 100

    def test_single_point(self):
        r = pd.Series([0.01])
        sr = sharpe_ratio(r, rf_annual=0.0)
        assert np.isnan(sr) or np.isinf(sr)


class TestAnnReturn:
    def test_zero_return(self):
        r = pd.Series(np.zeros(12))
        assert pytest.approx(ann_return(r)) == 0.0

    def test_one_percent_monthly(self):
        r = pd.Series(np.full(12, 0.01))
        assert pytest.approx(ann_return(r), abs=1e-6) == (1.01) ** 12 - 1

    def test_negative_monthly(self):
        r = pd.Series(np.full(12, -0.01))
        assert pytest.approx(ann_return(r), abs=1e-6) == (0.99) ** 12 - 1

    def test_single_month(self):
        r = pd.Series([0.02])
        assert pytest.approx(ann_return(r)) == (1.02) ** 12 - 1

    def test_mixed_returns(self):
        r = pd.Series([0.02, -0.01, 0.03])
        assert pytest.approx(ann_return(r)) == (1 + r.mean()) ** 12 - 1

    def test_empty_series(self):
        with pytest.raises(ValueError, match="need at least one array"):
            ann_return(pd.Series([], dtype=float))


class TestBuildBenchmarkFromFactors:
    def test_basic(self):
        ff = pd.DataFrame(
            {"Mkt-RF": [0.01, -0.005], "RF": [0.002, 0.002]},
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
        )
        bench = build_benchmark_from_factors(ff)
        assert pytest.approx(bench.iloc[0]) == 0.012
        assert pytest.approx(bench.iloc[1]) == -0.003

    def test_output_name(self):
        ff = pd.DataFrame(
            {"Mkt-RF": [0.01], "RF": [0.002]},
            index=pd.date_range("2020-01-31", periods=1, freq="ME"),
        )
        bench = build_benchmark_from_factors(ff)
        assert bench.name == "us_market"

    def test_freq_is_month_end(self):
        idx = pd.date_range("2020-01-15", "2020-03-15", freq="ME")
        ff = pd.DataFrame(
            {"Mkt-RF": [0.01, 0.02], "RF": [0.002, 0.002]},
            index=idx,
        )
        bench = build_benchmark_from_factors(ff)
        assert bench.index[0] == bench.index[0] + pd.offsets.MonthEnd(0)

    def test_with_nan_factors(self):
        ff = pd.DataFrame(
            {"Mkt-RF": [0.01, np.nan], "RF": [0.002, 0.002]},
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
        )
        bench = build_benchmark_from_factors(ff)
        assert pd.isna(bench.iloc[1])

    def test_alignment_with_missing_columns(self):
        ff = pd.DataFrame(
            {"Mkt-RF": [0.01]},
            index=pd.date_range("2020-01-31", periods=1, freq="ME"),
        )
        with pytest.raises(KeyError):
            build_benchmark_from_factors(ff)


class TestSummaryTable:
    def test_single_series(self):
        r = pd.Series(
            np.full(12, 0.01),
            index=pd.date_range("2020-01-31", periods=12, freq="ME"),
        )
        table = summary_table({"test": r})
        assert table.shape[0] == 1
        assert "Sharpe (2% rf)" in table.columns
        assert "Ann Return" in table.columns
        assert "Max DD" in table.columns

    def test_multiple_series(self):
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        r1 = pd.Series(np.full(12, 0.01), index=dates)
        r2 = pd.Series(np.full(12, 0.005), index=dates)
        table = summary_table({"a": r1, "b": r2})
        assert table.shape[0] == 2

    def test_known_values(self):
        r = pd.Series(
            np.full(24, 0.01),
            index=pd.date_range("2020-01-31", periods=24, freq="ME"),
        )
        table = summary_table({"s": r})
        row = table.loc["s"]
        assert pytest.approx(row["Mean (m)"]) == 0.01
        assert pytest.approx(row["Vol (m)"]) == 0.0
        assert pytest.approx(row["Ann Return"]) == (1.01) ** 12 - 1

    def test_nan_handling(self):
        r = pd.Series(
            [0.01, np.nan, 0.02],
            index=pd.date_range("2020-01-31", periods=3, freq="ME"),
        )
        table = summary_table({"s": r})
        assert table.loc["s", "N"] == 2

    def test_empty_series(self):
        r = pd.Series([], dtype=float)
        table = summary_table({"s": r})
        assert table.loc["s", "N"] == 0
        assert pd.isna(table.loc["s", "Mean (m)"])

    def test_index_name(self):
        r = pd.Series(
            np.full(12, 0.01),
            index=pd.date_range("2020-01-31", periods=12, freq="ME"),
        )
        table = summary_table({"test": r})
        assert table.index.name == "Series"
