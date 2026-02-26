"""Tests for targetviz.utils and targetviz.visualize modules."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


# ===================================================================
# utils.py
# ===================================================================


class TestGetTemplatePath:
    def test_returns_existing_path(self):
        from targetviz.utils import _get_template_path

        path = _get_template_path("base.html")
        # importlib.resources may return a Traversable; convert to string
        assert str(path).endswith("base.html")

    def test_extra_col_template(self):
        from targetviz.utils import _get_template_path

        path = _get_template_path("extra_col.html")
        assert str(path).endswith("extra_col.html")


class TestPlot360n0sc0pe:
    def test_returns_svg_string(self):
        from targetviz.utils import plot_360_n0sc0pe

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        svg = plot_360_n0sc0pe()
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_figure_is_closed(self):
        from targetviz.utils import plot_360_n0sc0pe

        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [1, 2])
        plot_360_n0sc0pe()
        # After calling, the figure should be closed — no open figures
        assert len(plt.get_fignums()) == 0

    def test_empty_plot(self):
        from targetviz.utils import plot_360_n0sc0pe

        fig, ax = plt.subplots()
        svg = plot_360_n0sc0pe()
        assert "<svg" in svg


class TestGetDfSmallEdgeCases:
    def test_empty_dataframe(self):
        from targetviz.utils import get_df_small

        df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
        result = get_df_small(df, "a", "b")
        assert len(result) == 0
        assert set(result.columns) == {"a", "b"}

    def test_returns_copy(self):
        from targetviz.utils import get_df_small

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = get_df_small(df, "x", "y")
        result["x"] = [99, 99]
        # Original should be untouched
        assert df["x"].tolist() == [1, 2]


# ===================================================================
# visualize.py
# ===================================================================


class TestPlotKde:
    def test_plots_without_error(self):
        from targetviz.config import Settings
        from targetviz.visualize import plot_kde

        config_ = Settings()
        s = pd.Series(np.random.randn(100))
        fig, ax = plt.subplots()
        plot_kde(s, ax, config_)
        plt.close()

    def test_samples_when_large(self):
        from targetviz.config import Settings
        from targetviz.visualize import plot_kde

        config_ = Settings()
        config_.kde.max_sample = 50
        s = pd.Series(np.random.randn(200))
        fig, ax = plt.subplots()
        # Should not raise
        plot_kde(s, ax, config_)
        plt.close()

    def test_constant_series_handles_linalg_error(self, caplog):
        """A constant series triggers LinAlgError which should be caught."""
        import logging

        from targetviz.config import Settings
        from targetviz.visualize import plot_kde

        config_ = Settings()
        s = pd.Series([5.0] * 100)
        fig, ax = plt.subplots()
        # Should not raise — LinAlgError is caught internally
        with caplog.at_level(logging.WARNING, logger="targetviz"):
            plot_kde(s, ax, config_)
        assert "Not able to compute KDE" in caplog.text
        plt.close()

    def test_small_series(self):
        from targetviz.config import Settings
        from targetviz.visualize import plot_kde

        config_ = Settings()
        config_.kde.max_sample = 10000
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fig, ax = plt.subplots()
        plot_kde(s, ax, config_)
        plt.close()


class TestTruncateLabels:
    def test_long_labels_are_truncated(self):
        from targetviz.config import Settings
        from targetviz.visualize import truncate_labels

        config_ = Settings()
        config_.max_lable_len = 10

        fig, ax = plt.subplots()
        ax.bar(["short", "a_very_very_long_label_here"], [1, 2])
        # Force rendering of tick labels
        fig.canvas.draw()

        truncate_labels(ax, config_)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label in labels:
            if label:  # skip empty labels
                assert len(label) <= 10

    def test_short_labels_unchanged(self):
        from targetviz.config import Settings
        from targetviz.visualize import truncate_labels

        config_ = Settings()
        config_.max_lable_len = 30

        fig, ax = plt.subplots()
        ax.bar(["abc", "def"], [1, 2])
        fig.canvas.draw()

        truncate_labels(ax, config_)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label in labels:
            if label:
                assert "..." not in label
        plt.close()

    def test_empty_plot(self):
        from targetviz.config import Settings
        from targetviz.visualize import truncate_labels

        config_ = Settings()
        fig, ax = plt.subplots()
        fig.canvas.draw()
        # Should not raise on empty plots
        truncate_labels(ax, config_)
        plt.close()
