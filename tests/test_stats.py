"""Tests for targetviz.stats module — covers all stats functions."""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper to build DescParams dicts
# ---------------------------------------------------------------------------


def _num_params(full_samp=True):
    return {"full_samp": full_samp, "is_cat": False, "is_date": False, "formatter": "{:.2f}"}


def _cat_params(full_samp=True):
    return {"full_samp": full_samp, "is_cat": True, "is_date": False, "formatter": "{:.2f}"}


def _date_params(full_samp=True):
    return {"full_samp": full_samp, "is_cat": False, "is_date": True, "formatter": "{}"}


# ===== get_median =====


class TestGetMedian:
    def test_numeric(self):
        from targetviz.stats import get_median

        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert get_median(s, _num_params()) == "3.00"

    def test_numeric_even_count(self):
        from targetviz.stats import get_median

        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert get_median(s, _num_params()) == "2.50"

    def test_categorical_returns_dash(self):
        from targetviz.stats import get_median

        s = pd.Series(["a", "b", "c"])
        assert get_median(s, _cat_params()) == "-"

    def test_date(self):
        from targetviz.stats import get_median

        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-01-03", "2020-01-05"]))
        result = get_median(s, _date_params())
        assert "2020" in str(result)

    def test_single_value(self):
        from targetviz.stats import get_median

        s = pd.Series([7.0])
        assert get_median(s, _num_params()) == "7.00"


# ===== get_mode =====


class TestGetMode:
    def test_numeric(self):
        from targetviz.stats import get_mode

        s = pd.Series([1.0, 2.0, 2.0, 3.0])
        assert get_mode(s, _num_params()) == "2.00"

    def test_categorical(self):
        from targetviz.stats import get_mode

        s = pd.Series(["a", "b", "a", "c"])
        assert get_mode(s, _cat_params()) == "a"

    def test_all_unique_returns_first(self):
        from targetviz.stats import get_mode

        s = pd.Series([10.0, 20.0, 30.0])
        # mode returns smallest when all unique
        result = get_mode(s, _num_params())
        assert result == "10.00"


# ===== get_std =====


class TestGetStd:
    def test_numeric(self):
        from targetviz.stats import get_std

        s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = get_std(s, _num_params())
        assert float(result) == pytest.approx(s.std(), abs=0.01)

    def test_categorical_returns_dash(self):
        from targetviz.stats import get_std

        s = pd.Series(["a", "b", "c"])
        assert get_std(s, _cat_params()) == "-"

    def test_date_returns_days_std(self):
        from targetviz.stats import get_std

        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-01-05", "2020-01-10"]))
        result = get_std(s, _date_params())
        # Should be the std of day-differences from the minimum
        assert float(result) > 0

    def test_constant_series(self):
        from targetviz.stats import get_std

        s = pd.Series([5.0, 5.0, 5.0])
        result = get_std(s, _num_params())
        assert float(result) == 0.0


# ===== get_quantiles =====


class TestGetQuantiles:
    def test_numeric(self):
        from targetviz.stats import get_quantiles

        s = pd.Series(range(101), dtype=float)
        result = get_quantiles(s, _num_params(), [0.25, 0.5, 0.75])
        assert len(result) == 3
        assert float(result[0]) == pytest.approx(25.0, abs=0.5)
        assert float(result[1]) == pytest.approx(50.0, abs=0.5)

    def test_categorical_returns_dashes(self):
        from targetviz.stats import get_quantiles

        s = pd.Series(["a", "b", "c"])
        result = get_quantiles(s, _cat_params(), [0.25, 0.5, 0.75])
        assert result == ["-", "-", "-"]

    def test_empty_quantile_list(self):
        from targetviz.stats import get_quantiles

        s = pd.Series([1.0, 2.0, 3.0])
        result = get_quantiles(s, _num_params(), [])
        assert result == []

    def test_single_quantile(self):
        from targetviz.stats import get_quantiles

        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = get_quantiles(s, _num_params(), [0.5])
        assert len(result) == 1
        assert float(result[0]) == pytest.approx(30.0, abs=0.5)


# ===== get_num_missing edge cases =====


class TestGetNumMissingEdgeCases:
    def test_all_nan(self):
        from targetviz.stats import get_num_missing

        s = pd.Series([np.nan, np.nan, np.nan])
        result = get_num_missing(s, _num_params(full_samp=True))
        assert result[0] == 3

    def test_no_nan(self):
        from targetviz.stats import get_num_missing

        s = pd.Series([1.0, 2.0, 3.0])
        result = get_num_missing(s, _num_params(full_samp=True))
        assert result[0] == 0
        assert "0.00%" in result[1]


# ===== get_min_max_mean edge cases =====


class TestGetMinMaxMeanEdgeCases:
    def test_single_value(self):
        from targetviz.stats import get_min_max_mean

        s = pd.Series([42.0])
        result = get_min_max_mean(s, _num_params())
        assert result == ["42.00", "42.00", "42.00"]

    def test_date_series(self):
        from targetviz.stats import get_min_max_mean

        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15", "2020-12-31"]))
        params = _date_params()
        # date is not cat, so should format
        result = get_min_max_mean(s, params)
        assert "2020" in str(result[0])

    def test_negative_values(self):
        from targetviz.stats import get_min_max_mean

        s = pd.Series([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = get_min_max_mean(s, _num_params())
        assert result == ["10.00", "-10.00", "0.00"]
