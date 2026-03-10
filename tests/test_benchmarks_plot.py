"""Pytest-benchmark tests for plot_kde and plot_histogram performance."""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from targetviz.analyzers import BaseAnalyzer  # noqa: E402
from targetviz.config import Settings  # noqa: E402
from targetviz.visualize import plot_kde  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_analyzer(config: Settings) -> BaseAnalyzer:
    """Create a BaseAnalyzer configured as a numeric column."""
    log = logging.getLogger("targetviz")
    analyzer = BaseAnalyzer(col="x", target="y", config_=config, log=log)
    analyzer.type = "NUM"
    return analyzer


def _make_cat_analyzer(config: Settings) -> BaseAnalyzer:
    """Create a BaseAnalyzer configured as a categorical column."""
    log = logging.getLogger("targetviz")
    analyzer = BaseAnalyzer(col="x", target="y", config_=config, log=log)
    analyzer.type = "CAT"
    return analyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return Settings()


@pytest.fixture
def small_numeric():
    """100 rows — tiny numeric series."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(100), name="x")


@pytest.fixture
def medium_numeric():
    """10 000 rows — typical numeric series."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(10_000), name="x")


@pytest.fixture
def large_numeric():
    """100 000 rows — large numeric series (exercises KDE sampling)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(100_000), name="x")


@pytest.fixture
def medium_categorical():
    """10 000 rows — categorical series with 15 categories."""
    rng = np.random.default_rng(42)
    cats = [f"cat_{i}" for i in range(15)]
    return pd.Series(
        pd.Categorical(rng.choice(cats, size=10_000)),
        name="x",
    )


@pytest.fixture
def binary_series():
    """10 000 rows — binary categorical series."""
    rng = np.random.default_rng(42)
    return pd.Series(
        pd.Categorical(rng.choice(["yes", "no"], size=10_000)),
        name="x",
    )


# ---------------------------------------------------------------------------
# Benchmarks — plot_kde
# ---------------------------------------------------------------------------


class TestPlotKdeBenchmark:
    """Benchmark KDE plotting across dataset sizes."""

    def test_kde_small(self, benchmark, config, small_numeric):
        def run():
            fig, ax = plt.subplots()
            plot_kde(small_numeric, ax, config)
            plt.close(fig)

        benchmark(run)

    def test_kde_medium(self, benchmark, config, medium_numeric):
        def run():
            fig, ax = plt.subplots()
            plot_kde(medium_numeric, ax, config)
            plt.close(fig)

        benchmark(run)

    def test_kde_large(self, benchmark, config, large_numeric):
        """Large series triggers sampling — should still be fast."""

        def run():
            fig, ax = plt.subplots()
            plot_kde(large_numeric, ax, config)
            plt.close(fig)

        benchmark(run)

    def test_kde_large_small_sample(self, benchmark, large_numeric):
        """KDE with a smaller max_sample to test sampling overhead."""
        config = Settings()
        config.kde.max_sample = 1000

        def run():
            fig, ax = plt.subplots()
            plot_kde(large_numeric, ax, config)
            plt.close(fig)

        benchmark(run)


# ---------------------------------------------------------------------------
# Benchmarks — plot_histogram (numeric)
# ---------------------------------------------------------------------------


class TestPlotHistogramNumericBenchmark:
    """Benchmark histogram + KDE for numeric columns."""

    def test_hist_small(self, benchmark, config, small_numeric):
        analyzer = _make_numeric_analyzer(config)

        def run():
            fig, ax = plt.subplots()
            analyzer.plot_histogram(small_numeric, ax)
            plt.close(fig)

        benchmark(run)

    def test_hist_medium(self, benchmark, config, medium_numeric):
        analyzer = _make_numeric_analyzer(config)

        def run():
            fig, ax = plt.subplots()
            analyzer.plot_histogram(medium_numeric, ax)
            plt.close(fig)

        benchmark(run)

    def test_hist_large(self, benchmark, config, large_numeric):
        analyzer = _make_numeric_analyzer(config)

        def run():
            fig, ax = plt.subplots()
            analyzer.plot_histogram(large_numeric, ax)
            plt.close(fig)

        benchmark(run)


# ---------------------------------------------------------------------------
# Benchmarks — plot_histogram (categorical / binary)
# ---------------------------------------------------------------------------


class TestPlotHistogramCategoricalBenchmark:
    """Benchmark histogram for categorical and binary columns."""

    def test_hist_categorical(self, benchmark, config, medium_categorical):
        analyzer = _make_cat_analyzer(config)

        def run():
            fig, ax = plt.subplots()
            analyzer.plot_histogram(medium_categorical, ax)
            plt.close(fig)

        benchmark(run)

    def test_hist_binary(self, benchmark, config, binary_series):
        log = logging.getLogger("targetviz")
        analyzer = BaseAnalyzer(col="x", target="y", config_=config, log=log)
        analyzer.type = "BINARY"

        def run():
            fig, ax = plt.subplots()
            analyzer.plot_histogram(binary_series, ax)
            plt.close(fig)

        benchmark(run)
