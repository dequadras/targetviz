"""Tests for targetviz.analyzers — dtype helpers, BaseAnalyzer, TargetAnalyzer, ColumnAnalyzer."""

from datetime import date, datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from targetviz.analyzers import (  # noqa: E402
    BaseAnalyzer,
    ColumnAnalyzer,
    TargetAnalyzer,
    _clean_fp_noise,
    _clean_interval_categories,
    _coerce_to_datetime,
    _is_date_object_column,
    _is_datetime_dtype,
    _is_numeric_dtype,
    _is_string_or_object_dtype,
)
from targetviz.config import Settings  # noqa: E402
from targetviz.utils import create_log  # noqa: E402


def _make_log():
    return create_log()


def _make_config():
    return Settings()


# ===================================================================
# Dtype helper functions
# ===================================================================


class TestIsStringOrObjectDtype:
    def test_object_dtype(self):
        assert _is_string_or_object_dtype(np.dtype("O")) is True

    def test_string_dtype(self):
        assert _is_string_or_object_dtype(pd.StringDtype()) is True

    def test_int_dtype(self):
        assert _is_string_or_object_dtype(np.dtype("int64")) is False

    def test_float_dtype(self):
        assert _is_string_or_object_dtype(np.dtype("float64")) is False

    def test_category_dtype(self):
        assert _is_string_or_object_dtype(pd.CategoricalDtype()) is False

    def test_bool_dtype(self):
        assert _is_string_or_object_dtype(np.dtype("bool")) is False

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_string(self):
        import pyarrow as pa

        assert _is_string_or_object_dtype(pd.ArrowDtype(pa.string())) is True

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_large_string(self):
        import pyarrow as pa

        assert _is_string_or_object_dtype(pd.ArrowDtype(pa.large_string())) is True

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_int_is_not_string(self):
        import pyarrow as pa

        assert _is_string_or_object_dtype(pd.ArrowDtype(pa.int64())) is False


class TestIsDatetimeDtype:
    def test_datetime64(self):
        assert _is_datetime_dtype(np.dtype("datetime64[ns]")) is True

    def test_datetime_tz(self):
        assert _is_datetime_dtype(pd.DatetimeTZDtype(tz="UTC")) is True

    def test_int_is_not_datetime(self):
        assert _is_datetime_dtype(np.dtype("int64")) is False

    def test_object_is_not_datetime(self):
        assert _is_datetime_dtype(np.dtype("O")) is False

    def test_string_is_not_datetime(self):
        assert _is_datetime_dtype(pd.StringDtype()) is False

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_timestamp(self):
        import pyarrow as pa

        assert _is_datetime_dtype(pd.ArrowDtype(pa.timestamp("ns"))) is True

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_date32(self):
        import pyarrow as pa

        assert _is_datetime_dtype(pd.ArrowDtype(pa.date32())) is True


class TestIsDateObjectColumn:
    def test_date_objects(self):
        s = pd.Series([date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)])
        assert _is_date_object_column(s) is True

    def test_datetime_objects(self):
        # pd.Series auto-converts datetime objects to datetime64[ns] dtype,
        # so we must force object dtype to test the date-object detection path.
        s = pd.Series([datetime(2020, 1, 1), datetime(2020, 6, 15)], dtype=object)
        assert _is_date_object_column(s) is True

    def test_string_objects(self):
        s = pd.Series(["2020-01-01", "2020-06-15"])
        assert _is_date_object_column(s) is False

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        assert _is_date_object_column(s) is False

    def test_all_nan(self):
        s = pd.Series([None, None, None])
        assert _is_date_object_column(s) is False

    def test_mixed_types(self):
        s = pd.Series([date(2020, 1, 1), "not a date", 42])
        assert _is_date_object_column(s) is False

    def test_non_object_dtype_returns_false(self):
        s = pd.Series([1, 2, 3])
        assert _is_date_object_column(s) is False


class TestCoerceToDatetime:
    def test_date_objects(self):
        s = pd.Series([date(2020, 1, 1), date(2020, 6, 15)])
        result = _coerce_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_already_datetime(self):
        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15"]))
        result = _coerce_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_string_dates(self):
        s = pd.Series(["2020-01-01", "2020-06-15"])
        result = _coerce_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_non_convertible_returns_original(self):
        s = pd.Series(["not a date", "also not"])
        result = _coerce_to_datetime(s)
        # Should return original if conversion fails
        assert len(result) == 2

    @pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="pyarrow not available")
    def test_pyarrow_date(self):
        import pyarrow as pa

        s = pd.Series(
            pd.array([date(2020, 1, 1), date(2020, 6, 15)], dtype=pd.ArrowDtype(pa.date32()))
        )
        result = _coerce_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)


class TestIsNumericDtype:
    def test_int64(self):
        assert _is_numeric_dtype(np.dtype("int64")) is True

    def test_float64(self):
        assert _is_numeric_dtype(np.dtype("float64")) is True

    def test_bool_excluded(self):
        assert _is_numeric_dtype(np.dtype("bool")) is False

    def test_object_excluded(self):
        assert _is_numeric_dtype(np.dtype("O")) is False

    def test_nullable_int(self):
        assert _is_numeric_dtype(pd.Int64Dtype()) is True

    def test_nullable_float(self):
        assert _is_numeric_dtype(pd.Float64Dtype()) is True

    def test_nullable_bool_excluded(self):
        assert _is_numeric_dtype(pd.BooleanDtype()) is False


# ===================================================================
# Floating-point noise cleanup helpers
# ===================================================================


class TestCleanFpNoise:
    """Tests for _clean_fp_noise."""

    def test_removes_trailing_nines_artefact(self):
        # Classic FP noise: 185.19899999999998 should become 185.199
        assert _clean_fp_noise(185.19899999999998) == 185.199

    def test_removes_trailing_zeros_artefact(self):
        assert _clean_fp_noise(384.7200000000001) == 384.72

    def test_preserves_meaningful_precision(self):
        # A value that genuinely needs 5 decimal places
        assert _clean_fp_noise(1.23456) == 1.23456

    def test_zero(self):
        assert _clean_fp_noise(0.0) == 0.0

    def test_negative(self):
        assert _clean_fp_noise(-185.19899999999998) == -185.199

    def test_integer_like_float(self):
        assert _clean_fp_noise(100.0) == 100.0

    def test_inf_unchanged(self):
        assert _clean_fp_noise(float("inf")) == float("inf")

    def test_nan_unchanged(self):
        import math

        assert math.isnan(_clean_fp_noise(float("nan")))

    def test_small_number(self):
        # 0.001 should stay as 0.001
        assert _clean_fp_noise(0.001) == 0.001


class TestCleanIntervalCategories:
    """Tests for _clean_interval_categories."""

    def test_cleans_interval_edges(self):
        # Simulate what pd.qcut would produce with FP noise
        idx = pd.IntervalIndex.from_breaks([185.19899999999998, 384.72, 600.0], closed="right")
        s = pd.Categorical.from_codes([0, 1, 0, 1], categories=idx)
        s = pd.Series(s)
        result = _clean_interval_categories(s)
        left_vals = [iv.left for iv in result.cat.categories]
        assert left_vals[0] == 185.199

    def test_non_interval_categories_unchanged(self):
        s = pd.Series(pd.Categorical(["a", "b", "a"]))
        result = _clean_interval_categories(s)
        assert list(result.cat.categories) == ["a", "b"]

    def test_already_clean_intervals_unchanged(self):
        idx = pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0], closed="right")
        s = pd.Series(pd.Categorical.from_codes([0, 1, 0], categories=idx))
        result = _clean_interval_categories(s)
        assert result.cat.categories.equals(idx)


# ===================================================================
# BaseAnalyzer.get_type
# ===================================================================


class TestBaseAnalyzerGetType:
    def _run_get_type(self, series, col="col", target="target"):
        config_ = _make_config()
        log = _make_log()
        ba = BaseAnalyzer(col, target, config_, log)
        df = pd.DataFrame({col: series, target: [0] * len(series)})
        df = ba.get_type(df)
        return ba.type, df

    def test_unique_column(self):
        typ, _ = self._run_get_type(pd.Series([5, 5, 5, 5]))
        assert typ == "UNIQUE"

    def test_binary_column(self):
        typ, _ = self._run_get_type(pd.Series([0, 1, 0, 1]))
        assert typ == "BINARY"

    def test_numeric_column(self):
        typ, _ = self._run_get_type(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert typ == "NUM"

    def test_string_column_becomes_cat(self):
        typ, _ = self._run_get_type(pd.Series(["a", "b", "c", "d"]))
        assert typ == "CAT"

    def test_category_column(self):
        typ, _ = self._run_get_type(pd.Series(pd.Categorical(["x", "y", "z", "x"])))
        assert typ == "CAT"

    def test_datetime_column(self):
        typ, _ = self._run_get_type(
            pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15", "2020-12-31"]))
        )
        assert typ == "DATE"

    def test_date_object_column(self):
        typ, _ = self._run_get_type(
            pd.Series([date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)])
        )
        assert typ == "DATE"

    def test_binary_datetime_becomes_date(self):
        """A column with exactly 2 datetime values should be DATE, not BINARY."""
        typ, _ = self._run_get_type(pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15"])))
        assert typ == "DATE"

    def test_bool_column_becomes_binary(self):
        typ, _ = self._run_get_type(pd.Series([True, False, True, False]))
        assert typ == "BINARY"


# ===================================================================
# TargetAnalyzer.check_types
# ===================================================================


class TestTargetAnalyzerCheckTypes:
    def test_valid_binary_target(self):
        config_ = _make_config()
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "BINARY"
        df = pd.DataFrame({"target": [0, 1, 0, 1]})
        ta.check_types(df)  # should not raise

    def test_valid_num_target(self):
        config_ = _make_config()
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "NUM"
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
        ta.check_types(df)  # should not raise

    def test_valid_cat_target(self):
        config_ = _make_config()
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "CAT"
        df = pd.DataFrame({"target": ["a", "b", "c"]})
        ta.check_types(df)

    def test_unique_target_raises(self):
        config_ = _make_config()
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "UNIQUE"
        df = pd.DataFrame({"target": [1, 1, 1]})
        with pytest.raises(TypeError, match="not an allowed type"):
            ta.check_types(df)

    def test_date_target_raises(self):
        config_ = _make_config()
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "DATE"
        df = pd.DataFrame({"target": pd.to_datetime(["2020-01-01", "2020-06-15"])})
        with pytest.raises(TypeError, match="not an allowed type"):
            ta.check_types(df)

    def test_too_many_cat_classes_raises(self):
        config_ = _make_config()
        config_.max_target_class = 3
        log = _make_log()
        ta = TargetAnalyzer("target", config_, log)
        ta.type = "CAT"
        df = pd.DataFrame({"target": ["a", "b", "c", "d", "e"]})
        with pytest.raises(AssertionError, match="too large"):
            ta.check_types(df)


# ===================================================================
# ColumnAnalyzer.clean_data
# ===================================================================


class TestColumnAnalyzerCleanData:
    def _make_analyzer(self, col="col", target="target", typ="NUM"):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer(col, target, config_, log)
        ca.type = typ
        return ca

    def test_removes_nan(self):
        ca = self._make_analyzer()
        ca.config.pct_outliers = 0.0  # disable outlier removal for this test
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0, 4.0], "target": [0, 1, 0, 1]})
        result = ca.clean_data(df)
        assert result["col"].isna().sum() == 0
        assert len(result) == 3

    def test_removes_infinite(self):
        ca = self._make_analyzer()
        df = pd.DataFrame({"col": [1.0, np.inf, -np.inf, 4.0], "target": [0, 1, 0, 1]})
        result = ca.clean_data(df)
        assert np.all(np.isfinite(result["col"].values))

    def test_rate_non_nulls(self):
        ca = self._make_analyzer()
        ca.config.pct_outliers = 0.0  # disable outlier removal for this test
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0, 4.0], "target": [0, 1, 0, 1]})
        ca.clean_data(df)
        assert ca.rate_non_nulls == pytest.approx(3 / 4)

    def test_removes_outliers_when_configured(self):
        ca = self._make_analyzer()
        ca.config.pct_outliers = 0.5  # aggressive outlier removal
        n = 100
        values = list(range(n))
        df = pd.DataFrame({"col": values, "target": [0] * n})
        result = ca.clean_data(df)
        # With 50% outlier removal, ~50% of rows should be kept
        assert len(result) < n

    def test_no_outlier_removal_when_zero(self):
        ca = self._make_analyzer()
        ca.config.pct_outliers = 0.0
        df = pd.DataFrame({"col": [1.0, 2.0, 100.0], "target": [0, 1, 0]})
        result = ca.clean_data(df)
        assert len(result) == 3

    def test_cat_type_skips_infinite_and_outlier_removal(self):
        ca = self._make_analyzer(typ="CAT")
        df = pd.DataFrame({"col": pd.Categorical(["a", "b", "c"]), "target": [0, 1, 0]})
        result = ca.clean_data(df)
        assert len(result) == 3


# ===================================================================
# ColumnAnalyzer.remove_outliers
# ===================================================================


class TestRemoveOutliers:
    def test_basic_removal(self):
        config_ = _make_config()
        config_.pct_outliers = 0.20
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        np.random.seed(42)
        df = pd.DataFrame({"col": np.arange(100, dtype=float), "target": [0] * 100})
        result = ca.remove_outliers(df)
        # 20% outliers = 10% from each tail
        assert len(result) < 100
        assert result["col"].min() >= df["col"].quantile(0.10)
        assert result["col"].max() <= df["col"].quantile(0.90)


# ===================================================================
# ColumnAnalyzer.sanity_checks
# ===================================================================


class TestSanityChecks:
    def test_passes_with_values(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": [1, 2, 3]})
        assert ca.sanity_checks(df) is True

    def test_fails_with_zero_unique(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": pd.Series([], dtype=float)})
        assert ca.sanity_checks(df) is False


# ===================================================================
# ColumnAnalyzer.change_types
# ===================================================================


class TestChangeTypes:
    def test_object_to_category(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        result = ca.change_types(df)
        assert result["col"].dtype.name == "category"

    def test_string_dtype_to_category(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": pd.array(["a", "b", "c"], dtype="string")})
        result = ca.change_types(df)
        assert result["col"].dtype.name == "category"

    def test_date_objects_not_converted_to_category(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)]})
        result = ca.change_types(df)
        # date objects should remain object dtype, NOT converted to category
        assert result["col"].dtype.name != "category"

    def test_numeric_unchanged(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        result = ca.change_types(df)
        assert pd.api.types.is_float_dtype(result["col"])


# ===================================================================
# ColumnAnalyzer.get_buckets
# ===================================================================


class TestGetBuckets:
    def _make_analyzer(self, col="col", target="target", typ="NUM"):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer(col, target, config_, log)
        ca.type = typ
        ca.rate_non_nulls = 1.0
        return ca

    def test_numeric_qcut(self):
        ca = self._make_analyzer()
        np.random.seed(42)
        df = pd.DataFrame({"col": np.random.randn(100), "target": np.random.randn(100)})
        result = ca.get_buckets(df)
        assert hasattr(result, "cat")  # categorical
        assert result.nunique() <= ca.config.n_breaks

    def test_cat_fewer_than_n_breaks(self):
        ca = self._make_analyzer(typ="CAT")
        ca.config.n_breaks = 10
        df = pd.DataFrame({"col": pd.Categorical(["a", "b", "c"] * 10), "target": [0] * 30})
        result = ca.get_buckets(df)
        assert set(result.unique()) == {"a", "b", "c"}

    def test_cat_more_than_n_breaks(self):
        ca = self._make_analyzer(typ="CAT")
        ca.config.n_breaks = 3
        categories = list("abcdefgh")
        df = pd.DataFrame(
            {"col": pd.Categorical(categories * 5), "target": [0] * len(categories) * 5}
        )
        result = ca.get_buckets(df)
        # Should have at most n_breaks categories (top 2 + "Other")
        assert result.nunique() <= 3

    def test_numeric_low_unique_values(self):
        ca = self._make_analyzer()
        df = pd.DataFrame({"col": [1.0, 2.0] * 10, "target": [0] * 20})
        result = ca.get_buckets(df)
        # With only 2 unique values, should be treated as category
        assert hasattr(result, "cat")

    def test_date_qcut(self):
        ca = self._make_analyzer(typ="DATE")
        dates = pd.to_datetime(pd.date_range("2020-01-01", periods=50, freq="D"))
        df = pd.DataFrame({"col": dates, "target": [0] * 50})
        result = ca.get_buckets(df)
        assert hasattr(result, "cat")


# ===================================================================
# ColumnAnalyzer.calc_explained_variance
# ===================================================================


class TestCalcExplainedVariance:
    def test_zero_variance_target(self):
        """When target has zero variance, explained variance should be 0."""
        result = ColumnAnalyzer.calc_explained_variance_(
            pd.Series([5.0, 5.0, 5.0, 5.0]),
            pd.Categorical(["a", "b", "a", "b"]),
        )
        assert result == 0.0

    def test_perfect_separation(self):
        """When buckets perfectly explain target, result should be ~1."""
        result = ColumnAnalyzer.calc_explained_variance_(
            pd.Series([0.0, 0.0, 1.0, 1.0]),
            pd.Categorical(["low", "low", "high", "high"]),
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_separation(self):
        """Same values in each bucket → explained variance ≈ 0."""
        result = ColumnAnalyzer.calc_explained_variance_(
            pd.Series([0.0, 1.0, 0.0, 1.0]),
            pd.Categorical(["a", "a", "b", "b"]),
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_calc_explained_variance_cat(self):
        """Test the categorical target path."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.rate_non_nulls = 1.0
        ca.config.target_type = "CAT"

        target = pd.Series(["a", "a", "b", "b", "a", "b"])
        cut = pd.Categorical(["low", "low", "low", "high", "high", "high"])
        result = ca.calc_explained_variance_cat(target, cut)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calc_explained_variance_num(self):
        """Test the numeric target path with rate_non_nulls weighting."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.rate_non_nulls = 0.5  # simulate 50% non-null

        target = pd.Series([0.0, 0.0, 1.0, 1.0])
        cut = pd.Categorical(["low", "low", "high", "high"])
        result = ca.calc_explained_variance_num(target, cut)
        # Should be exp_var * 0.5
        raw = ColumnAnalyzer.calc_explained_variance_(target, cut)
        assert result == pytest.approx(raw * 0.5)


# ===================================================================
# BaseAnalyzer.get_desc
# ===================================================================


class TestGetDesc:
    def test_numeric_full_sample(self):
        config_ = _make_config()
        log = _make_log()
        ba = BaseAnalyzer("col", "target", config_, log)
        ba.type = "NUM"
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ba.get_desc(s, full_samp=True, quantiles=[0.25, 0.75])
        # 10 base stats + 2 quantiles = 12
        assert len(result) == 12
        assert result[0] == 5  # num values

    def test_cat_series(self):
        config_ = _make_config()
        log = _make_log()
        ba = BaseAnalyzer("col", "target", config_, log)
        ba.type = "CAT"
        s = pd.Series(pd.Categorical(["a", "b", "a", "c"]))
        result = ba.get_desc(s, full_samp=True, quantiles=[0.25, 0.75])
        assert len(result) == 12
        assert result[0] == 4

    def test_date_series(self):
        config_ = _make_config()
        log = _make_log()
        ba = BaseAnalyzer("col", "target", config_, log)
        ba.type = "DATE"
        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15", "2020-12-31"]))
        result = ba.get_desc(s, full_samp=True, quantiles=[0.5])
        assert len(result) == 11  # 10 base stats + 1 quantile
        assert result[0] == 3

    def test_not_full_sample(self):
        config_ = _make_config()
        log = _make_log()
        ba = BaseAnalyzer("col", "target", config_, log)
        ba.type = "NUM"
        s = pd.Series([1.0, 2.0, 3.0])
        result = ba.get_desc(s, full_samp=False, quantiles=[0.5])
        # missing count/pct should be "-"
        assert result[2] == "-"
        assert result[3] == "-"


# ===================================================================
# ColumnAnalyzer._format_expected_value
# ===================================================================


class TestFormatExpectedValue:
    """Tests for _format_expected_value helper."""

    def test_numeric_target(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "NUM"
        result = ca._format_expected_value(pd.Series([10.0, 20.0, 30.0]))
        assert result == "20.0000"

    def test_binary_target(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "BINARY"
        result = ca._format_expected_value(pd.Series([0, 1, 1, 1]))
        assert "P(0)=25.00%" in result
        assert "P(1)=75.00%" in result

    def test_cat_target(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "CAT"
        result = ca._format_expected_value(pd.Series(["a", "b", "b", "c"]))
        assert "P(a)=25.00%" in result
        assert "P(b)=50.00%" in result
        assert "P(c)=25.00%" in result

    def test_cat_target_with_categorical_dtype(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "CAT"
        s = pd.Series(pd.Categorical(["x", "y", "x", "y", "y"]))
        result = ca._format_expected_value(s)
        assert "P(x)=40.00%" in result
        assert "P(y)=60.00%" in result

    def test_single_value_numeric(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "NUM"
        result = ca._format_expected_value(pd.Series([5.0]))
        assert result == "5.0000"

    def test_single_class_cat(self):
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.config.target_type = "CAT"
        result = ca._format_expected_value(pd.Series(["only"]))
        assert "P(only)=100.00%" in result


# ===================================================================
# ColumnAnalyzer.compute_target_expected_values
# ===================================================================


class TestComputeTargetExpectedValues:
    """Tests for compute_target_expected_values."""

    # --- Numeric target, numeric predictor ---

    def test_num_target_num_predictor_with_nulls_and_outliers(self):
        """Null/not-null and outlier/not-outlier stats present for NUM predictor."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0.05

        np.random.seed(0)
        col_vals = np.random.normal(50, 10, 200).tolist()
        target_vals = np.random.normal(100, 20, 200).tolist()
        # inject nulls
        col_vals[0] = np.nan
        col_vals[1] = np.nan
        col_vals[2] = np.nan
        df = pd.DataFrame({"col": col_vals, "target": target_vals})

        html = ca.compute_target_expected_values(df)
        assert "col is null" in html
        assert "col is not null" in html
        assert "col is outlier" in html
        assert "col is not outlier" in html
        assert "(n=3)" in html  # 3 nulls

    def test_num_target_no_nulls(self):
        """When no nulls exist, show 'No null values' message."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [10, 20, 30, 40, 50]})
        html = ca.compute_target_expected_values(df)
        assert "No null values" in html

    def test_no_outlier_section_for_cat_predictor(self):
        """Outlier analysis should NOT appear for categorical predictors."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "CAT"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame(
            {
                "col": pd.Categorical(["a", "b", "a", "b", np.nan]),
                "target": [10, 20, 30, 40, 50],
            }
        )
        html = ca.compute_target_expected_values(df)
        assert "col is null" in html
        assert "col is not null" in html
        assert "outlier" not in html

    def test_no_outlier_section_for_binary_predictor(self):
        """Outlier analysis should NOT appear for binary predictors."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "BINARY"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame(
            {
                "col": pd.Categorical([0, 1, 0, 1, np.nan]),
                "target": [10, 20, 30, 40, 50],
            }
        )
        html = ca.compute_target_expected_values(df)
        assert "col is null" in html
        assert "outlier" not in html

    # --- Binary target ---

    def test_binary_target_num_predictor(self):
        """Binary target should show class probabilities."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "BINARY"
        ca.config.pct_outliers = 0.05

        np.random.seed(1)
        n = 100
        col_vals = np.random.normal(0, 1, n).tolist()
        col_vals[0] = np.nan
        col_vals[1] = np.nan
        target_vals = np.random.choice([0, 1], size=n).tolist()
        df = pd.DataFrame({"col": col_vals, "target": target_vals})

        html = ca.compute_target_expected_values(df)
        assert "P(" in html
        assert "col is null" in html
        assert "col is not null" in html
        assert "col is outlier" in html
        assert "col is not outlier" in html

    def test_binary_target_cat_predictor(self):
        """Binary target + cat predictor: null/not-null but no outlier."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "CAT"
        ca.config.target_type = "BINARY"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame(
            {
                "col": pd.Categorical(["a", "b", "a", np.nan, "b"]),
                "target": [0, 1, 0, 1, 1],
            }
        )
        html = ca.compute_target_expected_values(df)
        assert "P(" in html
        assert "col is null" in html
        assert "col is not null" in html
        assert "outlier" not in html

    # --- Categorical target ---

    def test_cat_target_num_predictor(self):
        """CAT target should show per-class probabilities."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "CAT"
        ca.config.pct_outliers = 0.05

        np.random.seed(2)
        n = 100
        col_vals = np.random.normal(0, 1, n).tolist()
        col_vals[0] = np.nan
        target_vals = np.random.choice(["low", "mid", "high"], size=n).tolist()
        df = pd.DataFrame({"col": col_vals, "target": target_vals})

        html = ca.compute_target_expected_values(df)
        assert "P(" in html
        assert "col is null" in html
        assert "col is outlier" in html

    def test_cat_target_cat_predictor(self):
        """CAT target + CAT predictor: null/not-null, no outlier."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "CAT"
        ca.config.target_type = "CAT"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame(
            {
                "col": pd.Categorical(["x", "y", np.nan, "x", "y", "x"]),
                "target": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
            }
        )
        html = ca.compute_target_expected_values(df)
        assert "P(" in html
        assert "col is null" in html
        assert "col is not null" in html
        assert "outlier" not in html

    def test_cat_target_cat_predictor_no_nulls(self):
        """CAT target + CAT predictor with no nulls."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "CAT"
        ca.config.target_type = "CAT"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame(
            {
                "col": pd.Categorical(["x", "y", "x", "y"]),
                "target": pd.Categorical(["a", "b", "a", "b"]),
            }
        )
        html = ca.compute_target_expected_values(df)
        assert "No null values" in html
        assert "outlier" not in html

    # --- Edge cases ---

    def test_all_null_predictor(self):
        """When all predictor values are null, no null/not-null comparison possible."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0.05

        df = pd.DataFrame({"col": [np.nan, np.nan, np.nan], "target": [1, 2, 3]})
        html = ca.compute_target_expected_values(df)
        # All null → null_count > 0 but not_null_count == 0 → no null section
        assert "is null" not in html or "No null" in html

    def test_pct_outliers_zero_skips_outlier_section(self):
        """When pct_outliers=0, outlier analysis should be skipped."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0

        df = pd.DataFrame({"col": [1.0, 2.0, 3.0], "target": [10, 20, 30]})
        html = ca.compute_target_expected_values(df)
        assert "outlier" not in html

    def test_expected_value_correctness_numeric(self):
        """Check the actual numeric values are computed correctly."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "NUM"
        ca.config.pct_outliers = 0

        df = pd.DataFrame(
            {
                "col": [np.nan, np.nan, 1.0, 2.0],
                "target": [100.0, 200.0, 10.0, 20.0],
            }
        )
        html = ca.compute_target_expected_values(df)
        # E[target | col is null] = mean(100, 200) = 150.0
        assert "150.0000" in html
        # E[target | col is not null] = mean(10, 20) = 15.0
        assert "15.0000" in html

    def test_expected_value_correctness_binary_target(self):
        """Check class probabilities for binary target."""
        config_ = _make_config()
        log = _make_log()
        ca = ColumnAnalyzer("col", "target", config_, log)
        ca.type = "NUM"
        ca.config.target_type = "BINARY"
        ca.config.pct_outliers = 0

        df = pd.DataFrame(
            {
                "col": [np.nan, np.nan, 1.0, 2.0],
                "target": [0, 1, 1, 1],
            }
        )
        html = ca.compute_target_expected_values(df)
        # null group: [0, 1] → P(0)=50%, P(1)=50%
        assert "50.00%" in html
        # not-null group: [1, 1] → P(1)=100%
        assert "100.00%" in html
