"""Edge-case and error-path tests for targetviz end-to-end scenarios."""

import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from targetviz import targetviz_report  # noqa: E402


class TestEdgeCaseDataframes:
    """End-to-end tests with unusual or boundary DataFrames."""

    def test_single_feature_column(self):
        """DataFrame with only the target and one numeric feature."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_column_with_all_nan(self):
        """Column where feature is entirely NaN — should be skipped gracefully."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0],
                "good_col": [1.0, 2.0, 3.0, 4.0, 5.0],
                "nan_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_column_with_single_value(self):
        """Column with only one unique value — should be skipped as UNIQUE."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0],
                "const_col": [42, 42, 42, 42, 42],
                "var_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_date_feature_column(self):
        """DataFrame with a date feature column."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], 50),
                "date_col": dates,
                "num_col": np.random.randn(50),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_many_category_feature(self):
        """Categorical feature with many categories — triggers 'Other' bucketing."""
        np.random.seed(42)
        n = 200
        cats = [f"cat_{i}" for i in range(30)]
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "cat_col": np.random.choice(cats, n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_multiclass_cat_target(self):
        """Multiclass categorical target with numeric feature."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "target": np.random.choice(["low", "mid", "high"], n),
                "score": np.random.randn(n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_numeric_target_with_mixed_features(self):
        """Numeric target with a mix of numeric, categorical, and date features."""
        np.random.seed(42)
        n = 80
        df = pd.DataFrame(
            {
                "target": np.random.randn(n),
                "num_feature": np.random.randn(n),
                "cat_feature": np.random.choice(["a", "b", "c"], n),
                "date_feature": pd.date_range("2020-01-01", periods=n, freq="D"),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_feature_with_inf_values(self):
        """Numeric feature containing infinity values."""
        np.random.seed(42)
        values = np.random.randn(50)
        values[0] = np.inf
        values[1] = -np.inf
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], 50),
                "col": values,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_binary_feature_with_binary_target(self):
        """Both feature and target are binary."""
        np.random.seed(42)
        n = 60
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "binary_feat": np.random.choice(["yes", "no"], n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_boolean_feature(self):
        """Boolean dtype feature column."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "bool_col": np.random.choice([True, False], n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))


class TestKwargsPassthrough:
    """Verify kwargs are correctly passed to config."""

    def test_custom_n_breaks(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "x": np.random.randn(n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test.html",
                n_breaks=5,
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_custom_pct_outliers(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "x": np.random.randn(n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test.html",
                pct_outliers=0.0,
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))

    def test_custom_title(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "x": np.random.randn(n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test.html",
                title="Custom Title",
            )
            path = os.path.join(tmpdir, "test.html")
            assert os.path.exists(path)

    def test_invalid_kwarg_raises(self):
        df = pd.DataFrame({"target": [0, 1, 0], "x": [1, 2, 3]})
        with pytest.raises(ValueError):
            targetviz_report(
                df,
                target="target",
                totally_fake_param=42,
            )

    def test_nested_kwarg(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "x": np.random.randn(n),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test.html",
                hist={"max_values": 5},
            )
            assert os.path.exists(os.path.join(tmpdir, "test.html"))


class TestSkippedVariablesSummary:
    """Tests for the skipped-variables banner in the generated HTML report."""

    def test_skipped_constant_column_appears_in_report(self):
        """A constant column should be listed as skipped with correct reason."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1],
                "const_col": [7, 7, 7, 7, 7, 7],
                "good_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            html_path = os.path.join(tmpdir, "test.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            assert "1 variable skipped" in html
            assert "const_col" in html
            assert "Constant" in html

    def test_skipped_all_null_column_appears_in_report(self):
        """An all-null column should be listed as skipped."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1],
                "null_col": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "good_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            html_path = os.path.join(tmpdir, "test.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            assert "1 variable skipped" in html
            assert "null_col" in html
            assert "missing" in html

    def test_multiple_skipped_columns(self):
        """Multiple skipped columns should all appear."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1, 0, 1],
                "const_a": [1, 1, 1, 1, 1, 1, 1, 1],
                "null_b": [None] * 8,
                "const_c": ["x", "x", "x", "x", "x", "x", "x", "x"],
                "good_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            html_path = os.path.join(tmpdir, "test.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            assert "3 variables skipped" in html
            assert "const_a" in html
            assert "null_b" in html
            assert "const_c" in html

    def test_no_skipped_columns_no_banner(self):
        """When no columns are skipped, the banner should not appear."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [10, 20, 30, 40, 50, 60],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            html_path = os.path.join(tmpdir, "test.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            assert "variable" not in html.lower() or "skipped" not in html.lower()
            assert "<details" not in html

    def test_skipped_section_is_collapsed_by_default(self):
        """The details element should not have the 'open' attribute."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1],
                "const_col": [5, 5, 5, 5, 5, 5],
                "good_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df, target="target", output_dir=tmpdir + os.sep, name_file_out="test.html"
            )
            html_path = os.path.join(tmpdir, "test.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            # <details> present but NOT <details open>
            assert "<details" in html
            assert "open" not in html.split("</details>")[0].split("<details")[1]
