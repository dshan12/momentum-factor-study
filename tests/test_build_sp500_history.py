import pytest

from data.build_sp500_history import normalize_ticker, parse_date


class TestNormalizeTicker:
    def test_brk_b_dot(self):
        assert normalize_ticker("BRK.B") == "BRK-B"

    def test_brk_b_lowercase(self):
        assert normalize_ticker("brk.b") == "BRK-B"

    def test_bf_b_dot(self):
        assert normalize_ticker("BF.B") == "BF-B"

    def test_simple_ticker(self):
        assert normalize_ticker("AAPL") == "AAPL"

    def test_ticker_with_spaces(self):
        assert normalize_ticker("  MSFT  ") == "MSFT"

    def test_ticker_lowercase(self):
        assert normalize_ticker("aapl") == "AAPL"

    def test_ticker_with_dot(self):
        assert normalize_ticker("GOOG.L") == "GOOG-L"

    def test_nan_input(self):
        assert normalize_ticker(float("nan")) is None

    def test_none_input(self):
        assert normalize_ticker(None) is None

    def test_ticker_with_multiple_dots(self):
        assert normalize_ticker("A.B.C") == "A-B-C"

    def test_empty_string(self):
        assert normalize_ticker("") is None

    def test_already_normalized(self):
        assert normalize_ticker("BRK-B") == "BRK-B"


class TestParseDate:
    def test_iso_date(self):
        result = parse_date("2020-01-15")
        assert result == pd.Timestamp("2020-01-15")

    def test_mmddyyyy(self):
        result = parse_date("01/15/2020")
        assert result == pd.Timestamp("2020-01-15")

    def test_empty_string(self):
        result = parse_date("")
        assert result is pd.NaT

    def test_nan_input(self):
        result = parse_date(float("nan"))
        assert result is pd.NaT

    def test_invalid_string(self):
        result = parse_date("not-a-date")
        assert result is pd.NaT

    def test_month_year_format(self):
        result = parse_date("2020-01")
        assert result == pd.Timestamp("2020-01-31")

    def test_date_with_time(self):
        result = parse_date("2020-01-15 14:30:00")
        assert result == pd.Timestamp("2020-01-15")
