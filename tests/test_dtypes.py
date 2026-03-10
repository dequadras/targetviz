"""Tests for pandas dtype compatibility in targetviz.

Covers: numpy dtypes, pandas nullable types (Int/UInt/Float/boolean),
pyarrow-backed types, pandas StringDtype, and pandas 3 compatibility.
"""

import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


def _run_report(df, target="target", **kwargs):
    """Run targetviz_report inside a temp directory and assert output exists."""
    from targetviz import targetviz_report

    with tempfile.TemporaryDirectory() as tmpdir:
        targetviz_report(
            df,
            target=target,
            output_dir=tmpdir + os.sep,
            name_file_out="test_dtype.html",
            **kwargs,
        )
        assert any("test_dtype" in f for f in os.listdir(tmpdir))


N = 60  # sample size used across most tests


# ===========================================================================
# 1. Standard numpy dtypes
# ===========================================================================
class TestNumpyIntTypes:
    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_signed(self, dtype):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": np.array(np.random.randint(0, 50, N), dtype=dtype),
            }
        )
        _run_report(df)


class TestNumpyUintTypes:
    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
    def test_unsigned(self, dtype):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": np.array(np.random.randint(0, 50, N), dtype=dtype),
            }
        )
        _run_report(df)


class TestNumpyFloatTypes:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_float(self, dtype):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": np.array(np.random.randn(N), dtype=dtype),
            }
        )
        _run_report(df)


class TestNumpyBool:
    def test_bool(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": np.random.choice([True, False], N),
            }
        )
        _run_report(df)


# ===========================================================================
# 2. Pandas nullable integer types (capital-I: Int8 … Int64, UInt8 … UInt64)
# ===========================================================================
class TestNullableIntTypes:
    @pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64"])
    def test_nullable_int(self, dtype):
        vals = pd.array([1, 2, 3, pd.NA, 5, 6, 7, 8, 9, 10] * (N // 10), dtype=dtype)
        df = pd.DataFrame({"target": np.random.choice([0, 1], N), "col": vals})
        _run_report(df)


class TestNullableUIntTypes:
    @pytest.mark.parametrize("dtype", ["UInt8", "UInt16", "UInt32", "UInt64"])
    def test_nullable_uint(self, dtype):
        vals = pd.array([1, 2, 3, pd.NA, 5, 6, 7, 8, 9, 10] * (N // 10), dtype=dtype)
        df = pd.DataFrame({"target": np.random.choice([0, 1], N), "col": vals})
        _run_report(df)


# ===========================================================================
# 3. Pandas nullable float types (Float32, Float64)
# ===========================================================================
class TestNullableFloatTypes:
    @pytest.mark.parametrize("dtype", ["Float32", "Float64"])
    def test_nullable_float(self, dtype):
        vals = pd.array(
            [1.1, 2.2, 3.3, pd.NA, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0] * (N // 10), dtype=dtype
        )
        df = pd.DataFrame({"target": np.random.choice([0, 1], N), "col": vals})
        _run_report(df)


# ===========================================================================
# 4. Pandas nullable boolean type
# ===========================================================================
class TestNullableBoolType:
    def test_nullable_boolean(self):
        vals = pd.array([True, False, pd.NA, True, False] * (N // 5), dtype="boolean")
        df = pd.DataFrame({"target": np.random.choice([0, 1], N), "col": vals})
        _run_report(df)


# ===========================================================================
# 5. String types (object, StringDtype, pandas 3 default str)
# ===========================================================================
class TestStringTypes:
    def test_object_string(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": np.random.choice(["a", "b", "c", "d"], N),
            }
        )
        _run_report(df)

    def test_pandas_string_dtype(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(
                    np.random.choice(["cat", "dog", "bird", "fish"], N), dtype="string"
                ),
            }
        )
        _run_report(df)

    def test_string_as_target(self):
        """The target column itself can use StringDtype."""
        df = pd.DataFrame(
            {
                "target": pd.array(np.random.choice(["yes", "no"], N), dtype="string"),
                "col": np.random.randn(N),
            }
        )
        _run_report(df)


# ===========================================================================
# 6. PyArrow-backed types (pandas ≥ 2.0 + pyarrow installed)
# ===========================================================================
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")


class TestPyArrowNumeric:
    def test_int64(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(np.random.randint(0, 50, N), dtype="int64[pyarrow]"),
            }
        )
        _run_report(df)

    def test_uint32(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(
                    np.random.randint(0, 50, N).astype("uint32"), dtype="uint32[pyarrow]"
                ),
            }
        )
        _run_report(df)

    def test_float64(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(np.random.randn(N), dtype="float64[pyarrow]"),
            }
        )
        _run_report(df)

    def test_double(self):
        """pyarrow's float64 is named 'double' — make sure that works."""
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(np.random.randn(N), dtype="double[pyarrow]"),
            }
        )
        _run_report(df)


class TestPyArrowString:
    def test_string(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(
                    np.random.choice(["cat", "dog", "bird", "fish"], N),
                    dtype="string[pyarrow]",
                ),
            }
        )
        _run_report(df)

    def test_large_string(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(
                    np.random.choice(["alpha", "beta", "gamma"], N),
                    dtype=pd.ArrowDtype(pa.large_string()),
                ),
            }
        )
        _run_report(df)


class TestPyArrowBool:
    def test_bool(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(np.random.choice([True, False], N), dtype="bool[pyarrow]"),
            }
        )
        _run_report(df)


# ===========================================================================
# 7. Mixed dtype DataFrames
# ===========================================================================
class TestMixedTypes:
    def test_nullable_mix(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "int_col": pd.array(list(range(N)), dtype="Int64"),
                "uint_col": pd.array(list(range(N)), dtype="UInt32"),
                "float_col": pd.array(np.random.randn(N), dtype="Float64"),
                "str_col": pd.array(np.random.choice(["a", "b", "c"], N), dtype="string"),
                "bool_col": pd.array(np.random.choice([True, False], N), dtype="boolean"),
            }
        )
        _run_report(df)

    def test_pyarrow_mix(self):
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "int_col": pd.array(list(range(N)), dtype="int64[pyarrow]"),
                "float_col": pd.array(np.random.randn(N), dtype="double[pyarrow]"),
                "str_col": pd.array(np.random.choice(["x", "y", "z"], N), dtype="string[pyarrow]"),
            }
        )
        _run_report(df)

    def test_nullable_int_as_target(self):
        """Nullable integer target should work."""
        df = pd.DataFrame(
            {
                "target": pd.array([0, 1] * (N // 2), dtype="Int64"),
                "col": np.random.randn(N),
            }
        )
        _run_report(df)

    def test_nullable_float_as_target(self):
        """Nullable float target should work."""
        df = pd.DataFrame(
            {
                "target": pd.array(np.random.randn(N), dtype="Float64"),
                "col": np.random.randn(N),
            }
        )
        _run_report(df)


# ===========================================================================
# 8. Date / datetime types
# ===========================================================================
class TestDatetimeTypes:
    """Test various date and datetime representations."""

    def test_pandas_datetime(self):
        """Standard pd.to_datetime column."""
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.to_datetime(
                    pd.date_range("2020-01-01", periods=N, freq="D").to_series().values
                ),
            }
        )
        _run_report(df)

    def test_datetime64_ns(self):
        """Explicit datetime64[ns] dtype."""
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.date_range("2020-01-01", periods=N, freq="D"),
            }
        )
        _run_report(df)

    def test_datetime_with_timezone(self):
        """Timezone-aware datetime column."""
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.date_range("2020-01-01", periods=N, freq="D", tz="US/Eastern"),
            }
        )
        _run_report(df)

    def test_python_date_objects(self):
        """Column containing datetime.date objects (stored as object dtype)."""
        from datetime import date, timedelta

        base = date(2020, 1, 1)
        dates = [base + timedelta(days=i) for i in range(N)]
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": dates,
            }
        )
        assert df["col"].dtype == np.dtype("O"), "Pre-condition: date objects are object dtype"
        _run_report(df)

    def test_python_datetime_objects(self):
        """Column containing datetime.datetime objects (stored as object dtype)."""
        from datetime import datetime, timedelta

        base = datetime(2020, 1, 1, 12, 0)
        datetimes = [base + timedelta(days=i) for i in range(N)]
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": datetimes,
            }
        )
        _run_report(df)

    def test_date_with_missing(self):
        """Date column with NaT / None values."""
        dates = pd.date_range("2020-01-01", periods=N, freq="D").to_series().reset_index(drop=True)
        dates.iloc[5] = pd.NaT
        dates.iloc[15] = pd.NaT
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": dates,
            }
        )
        _run_report(df)


class TestPyArrowDateTypes:
    """Test pyarrow-backed date types."""

    pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")

    def test_date32(self):
        """PyArrow date32 type."""
        import pyarrow as pa

        dates = pd.date_range("2020-01-01", periods=N, freq="D")
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(dates, dtype=pd.ArrowDtype(pa.date32())),
            }
        )
        _run_report(df)

    def test_date64(self):
        """PyArrow date64 type."""
        import pyarrow as pa

        dates = pd.date_range("2020-01-01", periods=N, freq="D")
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(dates, dtype=pd.ArrowDtype(pa.date64())),
            }
        )
        _run_report(df)

    def test_timestamp_ns(self):
        """PyArrow timestamp[ns] type."""
        import pyarrow as pa

        dates = pd.date_range("2020-01-01", periods=N, freq="D")
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(dates, dtype=pd.ArrowDtype(pa.timestamp("ns"))),
            }
        )
        _run_report(df)

    def test_timestamp_us(self):
        """PyArrow timestamp[us] type."""
        import pyarrow as pa

        dates = pd.date_range("2020-01-01", periods=N, freq="D")
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], N),
                "col": pd.array(dates, dtype=pd.ArrowDtype(pa.timestamp("us"))),
            }
        )
        _run_report(df)
