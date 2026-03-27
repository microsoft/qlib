"""Test for Issue #1925: Freq normalization — '1day' vs 'day' equivalence."""
import pytest
from qlib.utils.resam import normalize_freq
from qlib.utils.time import Freq


class TestFreqNormalization:
    """Test that equivalent frequency strings normalize to the same canonical form."""

    def test_day_equivalents(self):
        assert normalize_freq("day") == normalize_freq("1day")
        assert normalize_freq("day") == normalize_freq("1d")
        assert normalize_freq("day") == normalize_freq("d")

    def test_min_equivalents(self):
        assert normalize_freq("min") == normalize_freq("1min")
        assert normalize_freq("min") == normalize_freq("minute")
        assert normalize_freq("min") == normalize_freq("1minute")

    def test_multi_count_preserved(self):
        assert normalize_freq("5min") == normalize_freq("5min")
        assert normalize_freq("5min") == normalize_freq("5minute")
        assert normalize_freq("30min") == "30min"

    def test_week_equivalents(self):
        assert normalize_freq("week") == normalize_freq("1week")
        assert normalize_freq("week") == normalize_freq("1w")
        assert normalize_freq("week") == normalize_freq("w")

    def test_month_equivalents(self):
        assert normalize_freq("month") == normalize_freq("1month")
        assert normalize_freq("month") == normalize_freq("1mon")
        assert normalize_freq("month") == normalize_freq("mon")

    def test_normalize_idempotent(self):
        """Normalizing an already-normalized string should return the same result."""
        for freq in ["day", "min", "5min", "week", "month", "30min"]:
            assert normalize_freq(freq) == normalize_freq(normalize_freq(freq))

    def test_freq_object_equality(self):
        """Freq objects from equivalent strings should be equal."""
        assert Freq("1day") == Freq("day")
        assert Freq("1min") == Freq("min")
        assert Freq("1d") == Freq("day")
        assert Freq("5min") == Freq("5minute")

    def test_canonical_strings(self):
        """Canonical output should use short forms."""
        assert normalize_freq("1day") == "day"
        assert normalize_freq("1min") == "1min"
        assert normalize_freq("5minute") == "5min"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
