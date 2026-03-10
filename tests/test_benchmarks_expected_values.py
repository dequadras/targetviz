"""Pytest-benchmark tests for ColumnAnalyzer.compute_target_expected_values."""

import logging

import numpy as np
import pandas as pd
import pytest

from targetviz.analyzers import ColumnAnalyzer
from targetviz.config import Settings
from targetviz.utils import create_log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> Settings:
    return Settings()


def _make_log() -> logging.Logger:
    return create_log()


def _make_analyzer(
    col: str = "col",
    target: str = "target",
    col_type: str = "NUM",
    target_type: str = "NUM",
    pct_outliers: float = 0.05,
) -> ColumnAnalyzer:
    config = _make_config()
    config.target_type = target_type
    config.pct_outliers = pct_outliers
    ca = ColumnAnalyzer(col, target, config, _make_log())
    ca.type = col_type
    return ca


# ---------------------------------------------------------------------------
# Fixtures — reusable DataFrames of various sizes
# ---------------------------------------------------------------------------


@pytest.fixture
def small_num_df():
    """200 rows, NUM predictor + NUM target, ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(200)
    target = rng.standard_normal(200) * 20 + 100
    col[rng.random(200) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


@pytest.fixture
def medium_num_df():
    """10 000 rows, NUM predictor + NUM target, ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(10_000)
    target = rng.standard_normal(10_000) * 20 + 100
    col[rng.random(10_000) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


@pytest.fixture
def large_num_df():
    """500 000 rows, NUM predictor + NUM target, ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(500_000)
    target = rng.standard_normal(500_000) * 20 + 100
    col[rng.random(500_000) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


@pytest.fixture
def medium_binary_df():
    """10 000 rows, NUM predictor + BINARY target, ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(10_000)
    target = rng.choice([0, 1], size=10_000)
    col[rng.random(10_000) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


@pytest.fixture
def medium_cat_df():
    """10 000 rows, NUM predictor + CAT target (3 classes), ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(10_000)
    target = rng.choice(["low", "mid", "high"], size=10_000)
    col[rng.random(10_000) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


@pytest.fixture
def large_binary_df():
    """500 000 rows, NUM predictor + BINARY target, ~5 % nulls."""
    rng = np.random.default_rng(42)
    col = rng.standard_normal(500_000)
    target = rng.choice([0, 1], size=500_000)
    col[rng.random(500_000) < 0.05] = np.nan
    return pd.DataFrame({"col": col, "target": target})


# ---------------------------------------------------------------------------
# Benchmarks — scaling with dataset size (NUM target)
# ---------------------------------------------------------------------------


class TestExpectedValuesBenchmarkBySize:
    """Benchmark compute_target_expected_values across different dataset sizes."""

    def test_small_num(self, benchmark, small_num_df):
        ca = _make_analyzer()
        html = benchmark(ca.compute_target_expected_values, small_num_df)
        assert "null" in html.lower()

    def test_medium_num(self, benchmark, medium_num_df):
        ca = _make_analyzer()
        html = benchmark(ca.compute_target_expected_values, medium_num_df)
        assert "null" in html.lower()

    def test_large_num(self, benchmark, large_num_df):
        ca = _make_analyzer()
        html = benchmark(ca.compute_target_expected_values, large_num_df)
        assert "null" in html.lower()


# ---------------------------------------------------------------------------
# Benchmarks — different target types
# ---------------------------------------------------------------------------


class TestExpectedValuesBenchmarkByTargetType:
    """Benchmark compute_target_expected_values for NUM, BINARY, and CAT targets."""

    def test_num_target(self, benchmark, medium_num_df):
        ca = _make_analyzer(target_type="NUM")
        html = benchmark(ca.compute_target_expected_values, medium_num_df)
        assert "outlier" in html.lower()

    def test_binary_target(self, benchmark, medium_binary_df):
        ca = _make_analyzer(target_type="BINARY")
        html = benchmark(ca.compute_target_expected_values, medium_binary_df)
        assert "P(" in html

    def test_cat_target(self, benchmark, medium_cat_df):
        ca = _make_analyzer(target_type="CAT")
        html = benchmark(ca.compute_target_expected_values, medium_cat_df)
        assert "P(" in html


# ---------------------------------------------------------------------------
# Benchmarks — with and without outlier detection
# ---------------------------------------------------------------------------


class TestExpectedValuesBenchmarkOutlierToggle:
    """Benchmark with outlier detection enabled vs disabled."""

    def test_with_outliers(self, benchmark, medium_num_df):
        ca = _make_analyzer(pct_outliers=0.05)
        html = benchmark(ca.compute_target_expected_values, medium_num_df)
        assert "outlier" in html.lower()

    def test_without_outliers(self, benchmark, medium_num_df):
        ca = _make_analyzer(pct_outliers=0.0)
        html = benchmark(ca.compute_target_expected_values, medium_num_df)
        assert "outlier" not in html.lower()


# ---------------------------------------------------------------------------
# Benchmarks — large dataset with BINARY target (stress test)
# ---------------------------------------------------------------------------


class TestExpectedValuesBenchmarkStress:
    """Stress test with 500k rows."""

    def test_large_binary(self, benchmark, large_binary_df):
        ca = _make_analyzer(target_type="BINARY")
        html = benchmark(ca.compute_target_expected_values, large_binary_df)
        assert "P(" in html

    def test_large_num(self, benchmark, large_num_df):
        ca = _make_analyzer(target_type="NUM")
        html = benchmark(ca.compute_target_expected_values, large_num_df)
        assert "outlier" in html.lower()


# ---------------------------------------------------------------------------
# Correctness under benchmark harness
# ---------------------------------------------------------------------------


class TestExpectedValuesCorrectnessUnderBenchmark:
    """Verify output correctness while benchmarking."""

    def test_no_nulls_message(self, benchmark):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0], "target": [10, 20, 30]})
        ca = _make_analyzer()
        html = benchmark(ca.compute_target_expected_values, df)
        assert "No null values" in html

    def test_null_counts_present(self, benchmark):
        col = [np.nan, np.nan, 1.0, 2.0, 3.0]
        target = [10, 20, 30, 40, 50]
        df = pd.DataFrame({"col": col, "target": target})
        ca = _make_analyzer()
        html = benchmark(ca.compute_target_expected_values, df)
        assert "(n=2)" in html  # 2 nulls
        assert "(n=3)" in html  # 3 non-nulls

    def test_cat_predictor_no_outliers(self, benchmark):
        df = pd.DataFrame(
            {
                "col": pd.Categorical(["a", "b", "a", np.nan, "b"]),
                "target": [10, 20, 30, 40, 50],
            }
        )
        ca = _make_analyzer(col_type="CAT")
        html = benchmark(ca.compute_target_expected_values, df)
        assert "outlier" not in html
