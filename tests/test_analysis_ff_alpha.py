import numpy as np
import pandas as pd
import pytest

from analysis_ff_alpha import load_factors, load_series, regress_excess, stars, to_ann


class TestToAnn:
    def test_zero(self):
        assert pytest.approx(to_ann(0.0)) == 0.0

    def test_one_percent(self):
        assert pytest.approx(to_ann(0.01), abs=1e-8) == (1.01) ** 12 - 1

    def test_two_percent(self):
        assert pytest.approx(to_ann(0.02), abs=1e-8) == (1.02) ** 12 - 1

    def test_negative(self):
        assert pytest.approx(to_ann(-0.01), abs=1e-8) == (0.99) ** 12 - 1

    def test_identity(self):
        assert pytest.approx(to_ann((1.10) ** (1 / 12) - 1), abs=1e-8) == 0.10


class TestStars:
    def test_p_below_1pct(self):
        assert stars(0.001) == "***"
        assert stars(0.009) == "***"

    def test_p_below_5pct(self):
        assert stars(0.01) == "**"
        assert stars(0.049) == "**"

    def test_p_below_10pct(self):
        assert stars(0.05) == "*"
        assert stars(0.099) == "*"

    def test_p_at_10pct(self):
        assert stars(0.10) == ""

    def test_p_above_10pct(self):
        assert stars(0.15) == ""
        assert stars(1.0) == ""

    def test_boundary_values(self):
        assert stars(0.00999) == "***"
        assert stars(0.01) == "**"
        assert stars(0.04999) == "**"
        assert stars(0.05) == "*"
        assert stars(0.09999) == "*"
        assert stars(0.10) == ""


class TestLoadSeries:
    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "series.csv"
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        pd.DataFrame({"date": dates, "value": [0.01, 0.02, -0.01]}).to_csv(
            csv, index=False
        )
        s = load_series(str(csv), "test_series")
        assert s.name == "test_series"
        assert len(s) == 3
        assert pytest.approx(s.iloc[0]) == 0.01

    def test_load_single_column_dataframe(self, tmp_path):
        csv = tmp_path / "series.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame({"date": dates, "val": [0.01, 0.02]}).set_index("date")
        df.to_csv(csv)
        s = load_series(str(csv), "test")
        assert isinstance(s, pd.Series)

    def test_output_freq(self, tmp_path):
        csv = tmp_path / "series.csv"
        dates = pd.date_range("2020-01-15", periods=3, freq="ME")
        pd.DataFrame({"date": dates, "value": [0.01, 0.02, 0.03]}).to_csv(
            csv, index=False
        )
        s = load_series(str(csv), "test")
        assert s.index.freqstr == "ME"

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "empty.csv"
        pd.DataFrame({"date": [], "value": []}).to_csv(csv, index=False)
        s = load_series(str(csv), "empty")
        assert len(s) == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_series("/nonexistent/path.csv", "x")


class TestRegressExcess:
    def test_regression_runs(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        cols = ["Mkt-RF", "SMB", "HML"]
        res = regress_excess(ret, fac, cols, lags=6)
        assert hasattr(res, "params")
        assert "const" in res.params
        assert res.nobs > 0

    def test_regression_output_structure(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        cols = ["Mkt-RF", "SMB"]
        res = regress_excess(ret, fac, cols, lags=6)
        for c in cols:
            assert c in res.params
        assert "const" in res.params
        assert hasattr(res, "rsquared")
        assert 0 <= res.rsquared <= 1

    def test_capm_single_factor(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        res = regress_excess(ret, fac, ["Mkt-RF"], lags=6)
        assert len(res.params) == 2
        assert "Mkt-RF" in res.params

    def test_ff5_all_factors(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        res = regress_excess(ret, fac, cols, lags=6)
        for c in cols:
            assert c in res.params

    def test_ff5_plus_umd(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
        res = regress_excess(ret, fac, cols, lags=6)
        for c in cols:
            assert c in res.params

    def test_regression_hac_uses_correct_lags(self, sample_returns, sample_factor_df):
        ret = sample_returns
        fac = sample_factor_df
        res_short = regress_excess(ret, fac, ["Mkt-RF"], lags=2)
        res_long = regress_excess(ret, fac, ["Mkt-RF"], lags=12)
        assert res_short.nobs == res_long.nobs


class TestLoadFactors:
    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "factors.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame(
            {
                "date": dates,
                "Mkt-RF": [0.01, -0.005],
                "SMB": [0.002, 0.001],
                "HML": [0.001, -0.001],
                "RMW": [0.002, 0.003],
                "CMA": [0.001, 0.002],
                "UMD": [0.004, -0.002],
                "RF": [0.002, 0.002],
            }
        )
        df.to_csv(csv, index=False)
        result = load_factors(str(csv))
        assert list(result.columns) == [
            "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"
        ]

    def test_missing_column_raises(self, tmp_path):
        csv = tmp_path / "factors.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame(
            {"date": dates, "Mkt-RF": [0.01, -0.005], "RF": [0.002, 0.002]}
        )
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Missing factor columns"):
            load_factors(str(csv))

    def test_output_freq(self, tmp_path):
        csv = tmp_path / "factors.csv"
        dates = pd.date_range("2020-01-15", periods=2, freq="ME")
        df = pd.DataFrame(
            {
                "date": dates, "Mkt-RF": [0.01, 0.02],
                "SMB": [0.0, 0.0], "HML": [0.0, 0.0],
                "RMW": [0.0, 0.0], "CMA": [0.0, 0.0],
                "UMD": [0.0, 0.0], "RF": [0.0, 0.0],
            }
        )
        df.to_csv(csv, index=False)
        result = load_factors(str(csv))
        assert result.index.freqstr == "ME"
