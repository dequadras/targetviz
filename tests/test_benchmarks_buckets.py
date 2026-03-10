"""Pytest-benchmark tests for get_buckets and calc_explained_variance_ performance."""

import logging

import numpy as np
import pandas as pd
import pytest

from targetviz.analyzers import ColumnAnalyzer
from targetviz.config import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return Settings()


@pytest.fixture
def log():
    return logging.getLogger("targetviz")


def _make_analyzer(col, target, config, log, col_type, rate_non_nulls=1.0):
    """Create a ColumnAnalyzer with the given type pre-set."""
    ca = ColumnAnalyzer(col, target, config, log)
    ca.type = col_type
    ca.rate_non_nulls = rate_non_nulls
    return ca


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df_10k():
    """10 000-row DataFrame with numeric column and numeric target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({"col": rng.standard_normal(10_000), "target": rng.standard_normal(10_000)})


@pytest.fixture
def numeric_df_100k():
    """100 000-row DataFrame with numeric column and numeric target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"col": rng.standard_normal(100_000), "target": rng.standard_normal(100_000)}
    )


@pytest.fixture
def date_df_10k():
    """10 000-row DataFrame with datetime column and numeric target."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=10_000, freq="h")
    return pd.DataFrame({"col": dates, "target": rng.standard_normal(10_000)})


@pytest.fixture
def date_df_100k():
    """100 000-row DataFrame with datetime column and numeric target."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100_000, freq="min")
    return pd.DataFrame({"col": dates, "target": rng.standard_normal(100_000)})


@pytest.fixture
def cat_df_10k():
    """10 000-row DataFrame with categorical column (20 categories) and numeric target."""
    rng = np.random.default_rng(42)
    cats = [f"cat_{i}" for i in range(20)]
    return pd.DataFrame(
        {
            "col": pd.Categorical(rng.choice(cats, size=10_000)),
            "target": rng.standard_normal(10_000),
        }
    )


@pytest.fixture
def binary_target_df_10k():
    """10 000-row DataFrame with numeric column and binary target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "col": rng.standard_normal(10_000),
            "target": pd.Categorical(rng.choice([0, 1], size=10_000)),
        }
    )


# ---------------------------------------------------------------------------
# Benchmarks — get_buckets by column type and size
# ---------------------------------------------------------------------------


class TestGetBucketsBenchmark:
    """Benchmark get_buckets across column types and sizes."""

    def test_numeric_10k(self, benchmark, config, log, numeric_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        result = benchmark(ca.get_buckets, numeric_df_10k)
        assert result.nunique() > 1

    def test_numeric_100k(self, benchmark, config, log, numeric_df_100k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        result = benchmark(ca.get_buckets, numeric_df_100k)
        assert result.nunique() > 1

    def test_date_10k(self, benchmark, config, log, date_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "DATE")
        result = benchmark(ca.get_buckets, date_df_10k)
        assert result.nunique() > 1

    def test_date_100k(self, benchmark, config, log, date_df_100k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "DATE")
        result = benchmark(ca.get_buckets, date_df_100k)
        assert result.nunique() > 1

    def test_categorical_10k(self, benchmark, config, log, cat_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "CAT")
        result = benchmark(ca.get_buckets, cat_df_10k)
        assert result.nunique() > 1


# ---------------------------------------------------------------------------
# Benchmarks — calc_explained_variance_ with pre-bucketed data
# ---------------------------------------------------------------------------


class TestExplainedVarianceBenchmark:
    """Benchmark calc_explained_variance_ across sizes."""

    def test_numeric_target_10k(self, benchmark, config, log, numeric_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        cut = pd.qcut(numeric_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_, numeric_df_10k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_numeric_target_100k(self, benchmark, config, log, numeric_df_100k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        cut = pd.qcut(numeric_df_100k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_, numeric_df_100k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_binary_target_10k(self, benchmark, config, log, binary_target_df_10k):
        config.target_type = "BINARY"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        cut = pd.qcut(binary_target_df_10k["col"], 10, duplicates="drop")
        target_binary = (binary_target_df_10k["target"] == 1).astype(float)
        result = benchmark(ca.calc_explained_variance_, target_binary, cut)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Benchmarks — calc_explained_variance_cat (vectorised crosstab path)
# ---------------------------------------------------------------------------


@pytest.fixture
def cat_target_df_10k():
    """10 000-row DataFrame with numeric column and 5-class categorical target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "col": rng.standard_normal(10_000),
            "target": pd.Categorical(rng.choice(["a", "b", "c", "d", "e"], size=10_000)),
        }
    )


@pytest.fixture
def cat_target_df_100k():
    """100 000-row DataFrame with numeric column and 5-class categorical target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "col": rng.standard_normal(100_000),
            "target": pd.Categorical(rng.choice(["a", "b", "c", "d", "e"], size=100_000)),
        }
    )


@pytest.fixture
def binary_target_df_100k():
    """100 000-row DataFrame with numeric column and binary target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "col": rng.standard_normal(100_000),
            "target": pd.Categorical(rng.choice([0, 1], size=100_000)),
        }
    )


class TestExplainedVarianceCatBenchmark:
    """Benchmark calc_explained_variance_cat across sizes and target cardinalities."""

    def test_binary_target_10k(self, benchmark, config, log, binary_target_df_10k):
        config.target_type = "BINARY"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(binary_target_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_cat, binary_target_df_10k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_binary_target_100k(self, benchmark, config, log, binary_target_df_100k):
        config.target_type = "BINARY"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(binary_target_df_100k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_cat, binary_target_df_100k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_5class_target_10k(self, benchmark, config, log, cat_target_df_10k):
        config.target_type = "CAT"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(cat_target_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_cat, cat_target_df_10k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_5class_target_100k(self, benchmark, config, log, cat_target_df_100k):
        config.target_type = "CAT"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(cat_target_df_100k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_cat, cat_target_df_100k["target"], cut)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Benchmarks — calc_explained_variance_num
# ---------------------------------------------------------------------------


class TestExplainedVarianceNumBenchmark:
    """Benchmark calc_explained_variance_num across sizes."""

    def test_numeric_10k(self, benchmark, config, log, numeric_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(numeric_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_num, numeric_df_10k["target"], cut)
        assert 0.0 <= result <= 1.0

    def test_numeric_100k(self, benchmark, config, log, numeric_df_100k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(numeric_df_100k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance_num, numeric_df_100k["target"], cut)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Benchmarks — full calc_explained_variance dispatch (NUM vs CAT/BINARY)
# ---------------------------------------------------------------------------


class TestExplainedVarianceDispatchBenchmark:
    """Benchmark the top-level calc_explained_variance method end-to-end."""

    def test_dispatch_num_10k(self, benchmark, config, log, numeric_df_10k):
        config.target_type = "NUM"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(numeric_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance, numeric_df_10k, cut)
        assert 0.0 <= result <= 1.0

    def test_dispatch_cat_10k(self, benchmark, config, log, cat_target_df_10k):
        config.target_type = "CAT"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(cat_target_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance, cat_target_df_10k, cut)
        assert 0.0 <= result <= 1.0

    def test_dispatch_binary_10k(self, benchmark, config, log, binary_target_df_10k):
        config.target_type = "BINARY"
        ca = _make_analyzer("col", "target", config, log, "NUM")
        ca.rate_non_nulls = 1.0
        cut = pd.qcut(binary_target_df_10k["col"], 10, duplicates="drop")
        result = benchmark(ca.calc_explained_variance, binary_target_df_10k, cut)
        assert 0.0 <= result <= 1.0
