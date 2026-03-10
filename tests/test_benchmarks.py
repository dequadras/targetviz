"""Pytest-benchmark tests for targetviz.stats — focused on get_quantiles."""

import numpy as np
import pandas as pd
import pytest

from targetviz.stats import get_quantiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUANTILES_3 = [0.25, 0.5, 0.75]
QUANTILES_7 = [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]


def _num_params():
    return {"full_samp": True, "is_cat": False, "is_date": False, "formatter": "{:.2f}"}


def _cat_params():
    return {"full_samp": True, "is_cat": True, "is_date": False, "formatter": "{:.2f}"}


# ---------------------------------------------------------------------------
# Fixtures — reusable series of various sizes
# ---------------------------------------------------------------------------


@pytest.fixture
def small_series():
    """100 rows — tiny dataset."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(100))


@pytest.fixture
def medium_series():
    """10 000 rows — typical dataset."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(10_000))


@pytest.fixture
def large_series():
    """1 000 000 rows — stress test."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(1_000_000))


@pytest.fixture
def series_with_nans():
    """10 000 rows, ~20 % NaN."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(10_000)
    mask = rng.random(10_000) < 0.2
    data[mask] = np.nan
    return pd.Series(data)


@pytest.fixture
def integer_series():
    """10 000 integer rows."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(0, 1000, size=10_000))


# ---------------------------------------------------------------------------
# Benchmarks — scaling with series size
# ---------------------------------------------------------------------------


class TestQuantileBenchmarkBySize:
    """Benchmark get_quantiles across different dataset sizes."""

    def test_small_3q(self, benchmark, small_series):
        result = benchmark(get_quantiles, small_series, _num_params(), QUANTILES_3)
        assert len(result) == 3

    def test_medium_3q(self, benchmark, medium_series):
        result = benchmark(get_quantiles, medium_series, _num_params(), QUANTILES_3)
        assert len(result) == 3

    def test_large_3q(self, benchmark, large_series):
        result = benchmark(get_quantiles, large_series, _num_params(), QUANTILES_3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Benchmarks — scaling with number of quantiles
# ---------------------------------------------------------------------------


class TestQuantileBenchmarkByQuantileCount:
    """Benchmark get_quantiles with different numbers of quantiles requested."""

    def test_medium_3q(self, benchmark, medium_series):
        result = benchmark(get_quantiles, medium_series, _num_params(), QUANTILES_3)
        assert len(result) == 3

    def test_medium_7q(self, benchmark, medium_series):
        result = benchmark(get_quantiles, medium_series, _num_params(), QUANTILES_7)
        assert len(result) == 7

    def test_medium_single_q(self, benchmark, medium_series):
        result = benchmark(get_quantiles, medium_series, _num_params(), [0.5])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Benchmarks — special data scenarios
# ---------------------------------------------------------------------------


class TestQuantileBenchmarkEdgeCases:
    """Benchmark get_quantiles under edge-case data conditions."""

    def test_with_nans(self, benchmark, series_with_nans):
        result = benchmark(get_quantiles, series_with_nans, _num_params(), QUANTILES_3)
        assert len(result) == 3
        assert all(r != "-" for r in result)

    def test_integer_data(self, benchmark, integer_series):
        result = benchmark(get_quantiles, integer_series, _num_params(), QUANTILES_3)
        assert len(result) == 3

    def test_categorical_short_circuits(self, benchmark, medium_series):
        """Categorical path should be near-instant (no computation)."""
        result = benchmark(get_quantiles, medium_series, _cat_params(), QUANTILES_3)
        assert result == ["-", "-", "-"]

    def test_empty_series(self, benchmark):
        s = pd.Series([], dtype=float)
        result = benchmark(get_quantiles, s, _num_params(), QUANTILES_3)
        assert len(result) == 3

    def test_constant_values(self, benchmark):
        s = pd.Series([42.0] * 10_000)
        result = benchmark(get_quantiles, s, _num_params(), QUANTILES_3)
        assert all(float(r) == pytest.approx(42.0) for r in result)


# ---------------------------------------------------------------------------
# Correctness checks (run inside benchmark harness for free regression)
# ---------------------------------------------------------------------------


class TestQuantileCorrectnessUnderBenchmark:
    """Verify numeric accuracy while benchmarking."""

    def test_known_values(self, benchmark):
        s = pd.Series(range(101), dtype=float)  # 0..100
        result = benchmark(get_quantiles, s, _num_params(), QUANTILES_3)
        assert float(result[0]) == pytest.approx(25.0, abs=0.5)
        assert float(result[1]) == pytest.approx(50.0, abs=0.5)
        assert float(result[2]) == pytest.approx(75.0, abs=0.5)

    def test_quantiles_are_monotonic(self, benchmark, large_series):
        result = benchmark(get_quantiles, large_series, _num_params(), QUANTILES_7)
        values = [float(r) for r in result]
        assert values == sorted(values), "Quantile results must be monotonically non-decreasing"

    def test_median_matches_numpy(self, benchmark, medium_series):
        result = benchmark(get_quantiles, medium_series, _num_params(), [0.5])
        expected = np.nanmedian(medium_series.to_numpy())
        assert float(result[0]) == pytest.approx(expected, abs=0.01)
