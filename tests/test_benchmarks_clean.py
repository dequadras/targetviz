"""Pytest-benchmark tests for ColumnAnalyzer.clean_data and remove_outliers."""

import logging

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from targetviz.analyzers import ColumnAnalyzer  # noqa: E402
from targetviz.config import Settings  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analyzer(col="col", target="target", typ="NUM", pct_outliers=0.05):
    """Create a ColumnAnalyzer ready for benchmarking."""
    config_ = Settings()
    config_.pct_outliers = pct_outliers
    log = logging.getLogger("targetviz")
    ca = ColumnAnalyzer(col, target, config_, log)
    ca.type = typ
    return ca


def _make_numeric_df(n, rng, nan_frac=0.0, inf_frac=0.0):
    """Build a DataFrame with numeric col + target, optional NaN/Inf."""
    values = rng.standard_normal(n)
    if nan_frac > 0:
        mask = rng.random(n) < nan_frac
        values[mask] = np.nan
    if inf_frac > 0:
        mask = rng.random(n) < inf_frac
        values[mask] = rng.choice([np.inf, -np.inf], size=mask.sum())
    return pd.DataFrame({"col": values, "target": rng.integers(0, 2, size=n)})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    """100 rows — tiny dataset."""
    return _make_numeric_df(100, np.random.default_rng(42))


@pytest.fixture
def medium_df():
    """10 000 rows — typical dataset."""
    return _make_numeric_df(10_000, np.random.default_rng(42))


@pytest.fixture
def large_df():
    """500 000 rows — stress test."""
    return _make_numeric_df(500_000, np.random.default_rng(42))


@pytest.fixture
def medium_df_dirty():
    """10 000 rows with ~10% NaN and ~5% Inf."""
    return _make_numeric_df(10_000, np.random.default_rng(42), nan_frac=0.10, inf_frac=0.05)


@pytest.fixture
def large_df_dirty():
    """500 000 rows with ~10% NaN and ~5% Inf."""
    return _make_numeric_df(500_000, np.random.default_rng(42), nan_frac=0.10, inf_frac=0.05)


# ---------------------------------------------------------------------------
# Benchmarks — clean_data scaling with size
# ---------------------------------------------------------------------------


class TestCleanDataBenchmarkBySize:
    """Benchmark clean_data across different dataset sizes."""

    def test_small(self, benchmark, small_df):
        ca = _make_analyzer()
        result = benchmark(ca.clean_data, small_df)
        assert len(result) <= len(small_df)

    def test_medium(self, benchmark, medium_df):
        ca = _make_analyzer()
        result = benchmark(ca.clean_data, medium_df)
        assert len(result) <= len(medium_df)

    def test_large(self, benchmark, large_df):
        ca = _make_analyzer()
        result = benchmark(ca.clean_data, large_df)
        assert len(result) <= len(large_df)


# ---------------------------------------------------------------------------
# Benchmarks — clean_data with dirty data (NaN + Inf)
# ---------------------------------------------------------------------------


class TestCleanDataBenchmarkDirty:
    """Benchmark clean_data with NaN and Inf values."""

    def test_medium_dirty(self, benchmark, medium_df_dirty):
        ca = _make_analyzer()
        result = benchmark(ca.clean_data, medium_df_dirty)
        assert len(result) < len(medium_df_dirty)
        assert result["col"].isna().sum() == 0

    def test_large_dirty(self, benchmark, large_df_dirty):
        ca = _make_analyzer()
        result = benchmark(ca.clean_data, large_df_dirty)
        assert len(result) < len(large_df_dirty)
        assert result["col"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Benchmarks — remove_outliers isolation
# ---------------------------------------------------------------------------


class TestRemoveOutliersBenchmark:
    """Benchmark remove_outliers in isolation (no NaN/Inf, just outlier trimming)."""

    def test_medium(self, benchmark, medium_df):
        ca = _make_analyzer(pct_outliers=0.05)
        result = benchmark(ca.remove_outliers, medium_df)
        assert len(result) < len(medium_df)

    def test_large(self, benchmark, large_df):
        ca = _make_analyzer(pct_outliers=0.05)
        result = benchmark(ca.remove_outliers, large_df)
        assert len(result) < len(large_df)

    def test_aggressive_outliers(self, benchmark, medium_df):
        """50% outlier removal — worst case for quantile computation."""
        ca = _make_analyzer(pct_outliers=0.50)
        result = benchmark(ca.remove_outliers, medium_df)
        assert len(result) < len(medium_df)


# ---------------------------------------------------------------------------
# Benchmarks — clean_data with outlier removal disabled
# ---------------------------------------------------------------------------


class TestCleanDataNoOutliers:
    """Benchmark clean_data with pct_outliers=0 to isolate NaN/Inf filtering."""

    def test_medium_no_outliers(self, benchmark, medium_df_dirty):
        ca = _make_analyzer(pct_outliers=0.0)
        result = benchmark(ca.clean_data, medium_df_dirty)
        assert len(result) < len(medium_df_dirty)

    def test_large_no_outliers(self, benchmark, large_df_dirty):
        ca = _make_analyzer(pct_outliers=0.0)
        result = benchmark(ca.clean_data, large_df_dirty)
        assert len(result) < len(large_df_dirty)


# ---------------------------------------------------------------------------
# Benchmarks — categorical path (should skip numeric-only work)
# ---------------------------------------------------------------------------


class TestCleanDataCategorical:
    """Benchmark clean_data on categorical data — should skip Inf/outlier logic."""

    def test_medium_cat(self, benchmark):
        rng = np.random.default_rng(42)
        cats = rng.choice(["a", "b", "c", "d", "e"], size=10_000)
        df = pd.DataFrame(
            {
                "col": pd.Categorical(cats),
                "target": rng.integers(0, 2, size=10_000),
            }
        )
        ca = _make_analyzer(typ="CAT")
        result = benchmark(ca.clean_data, df)
        assert len(result) == 10_000
