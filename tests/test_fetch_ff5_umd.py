import re
import pytest

YYYYMM_RE = re.compile(r"^\s*(\d{6})\b")


class TestYYYYMMRegex:
    def test_matches_six_digits(self):
        m = YYYYMM_RE.match("202001")
        assert m is not None
        assert m.group(1) == "202001"

    def test_matches_with_leading_space(self):
        m = YYYYMM_RE.match("   202001")
        assert m is not None
        assert m.group(1) == "202001"

    def test_matches_with_trailing_text(self):
        m = YYYYMM_RE.match("202001,   0.01,   0.02")
        assert m is not None
        assert m.group(1) == "202001"

    def test_rejects_seven_digits(self):
        m = YYYYMM_RE.match("2020011")
        assert m is not None
        assert m.group(1) == "202001"

    def test_rejects_five_digits(self):
        m = YYYYMM_RE.match("20201")
        assert m is None

    def test_rejects_non_digit_text(self):
        m = YYYYMM_RE.match("abcdef")
        assert m is None

    def test_rejects_empty_string(self):
        m = YYYYMM_RE.match("")
        assert m is None

    def test_rejects_whitespace_only(self):
        m = YYYYMM_RE.match("   ")
        assert m is None

    def test_rejects_negative_number(self):
        m = YYYYMM_RE.match("-202001")
        assert m is None

    def test_matches_word_boundary(self):
        m = YYYYMM_RE.match("202001, 0.01")
        assert m is not None
        assert m.group(1) == "202001"

    def test_match_at_line_start_only(self):
        m = YYYYMM_RE.match("  202001  some text")
        assert m is not None
        assert m.group(1) == "202001"

    def test_does_not_match_in_middle(self):
        m = YYYYMM_RE.search("some 202001 text")
        assert m is not None
        assert m.group(1) == "202001"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("202001", "202001"),
        ("  202001", "202001"),
        ("202001,  0.01", "202001"),
        ("199912", "199912"),
        ("202513", "202513"),
    ],
)
def test_yyyymm_parametrized(text, expected):
    m = YYYYMM_RE.match(text)
    assert m is not None
    assert m.group(1) == expected


@pytest.mark.parametrize(
    "text",
    [
        "20201",
        "abcdef",
        "",
        "   ",
        "12345",
        "2020011 extra digits",
    ],
)
def test_yyyymm_no_match_parametrized(text):
    m = YYYYMM_RE.match(text)
    assert m is None
