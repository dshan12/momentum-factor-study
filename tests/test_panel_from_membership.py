import numpy as np
import pandas as pd
import pytest

from data.panel_from_membership import (
    load_membership_monthly,
    load_prices_union,
    prices_masked_by_membership,
)


class TestLoadPricesUnion:
    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "prices.csv"
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        df = pd.DataFrame(
            {"date": dates, "AAPL": [100.0, 105.0, 102.0], "MSFT": [200.0, 198.0, 210.0]}
        )
        df.to_csv(csv, index=False)

        result = load_prices_union(str(csv))
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(dates)
        assert list(result.columns) == ["AAPL", "MSFT"]
        assert pytest.approx(result.loc[dates[0], "AAPL"]) == 100.0

    def test_load_sorts_by_date(self, tmp_path):
        csv = tmp_path / "prices.csv"
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        df = pd.DataFrame(
            {
                "date": dates[::-1],
                "AAPL": [102.0, 105.0, 100.0],
            }
        )
        df.to_csv(csv, index=False)

        result = load_prices_union(str(csv))
        assert result.index[0] == dates[0]
        assert result.index[-1] == dates[-1]

    def test_column_names_are_strings(self, tmp_path):
        csv = tmp_path / "prices.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame(
            {"date": dates, "AAPL": [100.0, 105.0]}
        )
        df.to_csv(csv, index=False)

        result = load_prices_union(str(csv))
        assert all(isinstance(c, str) for c in result.columns)

    def test_index_is_month_end(self, tmp_path):
        csv = tmp_path / "prices.csv"
        dates = pd.date_range("2020-01-15", periods=2, freq="ME")
        df = pd.DataFrame(
            {"date": dates, "AAPL": [100.0, 105.0]}
        )
        df.to_csv(csv, index=False)

        result = load_prices_union(str(csv))
        for d in result.index:
            assert d == d + pd.offsets.MonthEnd(0)

    def test_single_ticker_csv(self, tmp_path):
        csv = tmp_path / "prices.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame({"date": dates, "AAPL": [100.0, 105.0]})
        df.to_csv(csv, index=False)
        result = load_prices_union(str(csv))
        assert result.shape[1] == 1
        assert result.columns[0] == "AAPL"

    def test_empty_date_raises(self, tmp_path):
        csv = tmp_path / "prices.csv"
        pd.DataFrame({"date": [], "AAPL": []}).to_csv(csv, index=False)
        result = load_prices_union(str(csv))
        assert len(result) == 0


class TestLoadMembershipMonthly:
    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "membership.csv"
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame(
            {
                "date": [dates[0], dates[0], dates[1], dates[1]],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "in_index": [1, 1, 1, 1],
            }
        )
        df.to_csv(csv, index=False)

        result = load_membership_monthly(str(csv))
        assert len(result) == 4
        assert list(result.columns) == ["date", "ticker", "in_index"]

    def test_ticker_dtype_is_str(self, tmp_path):
        csv = tmp_path / "membership.csv"
        dates = pd.date_range("2020-01-31", periods=1, freq="ME")
        df = pd.DataFrame(
            {"date": [dates[0]], "ticker": ["123"], "in_index": [1]}
        )
        df.to_csv(csv, index=False)
        result = load_membership_monthly(str(csv))
        assert result["ticker"].dtype == "object" or result["ticker"].dtype == "string"


class TestPricesMaskedByMembership:
    def test_basic_masking(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        prices = pd.DataFrame(
            {"AAPL": [100.0, 105.0, 102.0], "MSFT": [200.0, 198.0, 210.0]},
            index=dates,
        )
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
                "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
                "in_index": [1, 1, 1, 1, 1, 1],
            }
        )

        masked = prices_masked_by_membership(prices, membership)
        assert not masked.isna().any().any()
        assert pytest.approx(masked.loc[dates[0], "AAPL"]) == 100.0

    def test_masks_out_non_members(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        prices = pd.DataFrame(
            {"AAPL": [100.0, 105.0], "MSFT": [200.0, 198.0]},
            index=dates,
        )
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "ticker": ["AAPL", "MSFT"],
                "in_index": [1, 0],
            }
        )
        masked = prices_masked_by_membership(prices, membership)
        assert not np.isnan(masked.loc[dates[0], "AAPL"])
        assert np.isnan(masked.loc[dates[0], "MSFT"])

    def test_no_overlapping_tickers_raises(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        prices = pd.DataFrame({"AAPL": [100.0, 105.0]}, index=dates)
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[1]],
                "ticker": ["MSFT", "MSFT"],
                "in_index": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="No overlapping tickers"):
            prices_masked_by_membership(prices, membership)

    def test_prices_superset_of_membership(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        prices = pd.DataFrame(
            {"AAPL": [100.0, 105.0], "GOOG": [200.0, 210.0]},
            index=dates,
        )
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "ticker": ["AAPL", "AAPL"],
                "in_index": [1, 1],
            }
        )
        masked = prices_masked_by_membership(prices, membership)
        assert list(masked.columns) == ["AAPL"]
        assert "GOOG" not in masked.columns

    def test_membership_with_extra_tickers_ignored(self):
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        prices = pd.DataFrame({"AAPL": [100.0, 105.0]}, index=dates)
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "ticker": ["AAPL", "MSFT"],
                "in_index": [1, 1],
            }
        )
        masked = prices_masked_by_membership(prices, membership)
        assert list(masked.columns) == ["AAPL"]

    def test_date_alignment_different_order(self):
        dates = pd.date_range("2020-03-31", periods=2, freq="ME")
        prices = pd.DataFrame({"AAPL": [105.0, 102.0]}, index=dates)
        membership_dates = pd.date_range("2020-01-31", periods=4, freq="ME")
        membership = pd.DataFrame(
            {
                "date": list(membership_dates) * 2,
                "ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
                "in_index": [1, 1, 0, 1] + [1, 1, 1, 1],
            }
        )
        masked = prices_masked_by_membership(prices, membership)
        assert masked.loc[dates[0], "AAPL"] == 105.0

    def test_missing_membership_date_fills_nan(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        prices = pd.DataFrame(
            {"AAPL": [100.0, 105.0, 102.0]}, index=dates
        )
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[2]],
                "ticker": ["AAPL", "AAPL"],
                "in_index": [1, 1],
            }
        )
        masked = prices_masked_by_membership(prices, membership)
        assert not np.isnan(masked.loc[dates[0], "AAPL"])
        assert np.isnan(masked.loc[dates[1], "AAPL"])
        assert not np.isnan(masked.loc[dates[2], "AAPL"])
