"""Tests verifying the modularized targetviz package works correctly."""

import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


class TestImports:
    """Verify all modules import correctly and have no circular dependencies."""

    def test_import_typedefs(self):
        from targetviz import typedefs  # noqa: F401

    def test_import_utils(self):
        from targetviz import utils  # noqa: F401

    def test_import_stats(self):
        from targetviz import stats  # noqa: F401

    def test_import_visualize(self):
        from targetviz import visualize  # noqa: F401

    def test_import_analyzers(self):
        from targetviz import analyzers  # noqa: F401

    def test_import_report(self):
        from targetviz import report  # noqa: F401

    def test_import_profile_report(self):
        from targetviz import profile_report  # noqa: F401

    def test_import_public_api(self):
        from targetviz import targetviz_report  # noqa: F401


class TestUtils:
    def test_get_df_small_same_col(self):
        from targetviz.utils import get_df_small

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_df_small(df, "a", "a")
        assert list(result.columns) == ["a"]

    def test_get_df_small_diff_col(self):
        from targetviz.utils import get_df_small

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_df_small(df, "a", "b")
        assert set(result.columns) == {"a", "b"}

    def test_create_log(self):
        from targetviz.utils import create_log

        log = create_log()
        assert log.name == "targetviz"


class TestStats:
    def test_get_num_values(self):
        from targetviz.stats import get_num_values

        s = pd.Series([1, 2, 3, 4, 5])
        assert get_num_values(s) == 5

    def test_get_num_unique_values(self):
        from targetviz.stats import get_num_unique_values

        s = pd.Series([1, 1, 2, 2, 3])
        assert get_num_unique_values(s) == 3

    def test_get_num_missing_full_sample(self):
        from targetviz.stats import get_num_missing

        s = pd.Series([1, np.nan, 3])
        params = {"full_samp": True, "is_cat": False, "is_date": False, "formatter": "{:.2f}"}
        result = get_num_missing(s, params)
        assert result[0] == 1

    def test_get_num_missing_not_full_sample(self):
        from targetviz.stats import get_num_missing

        s = pd.Series([1, 2, 3])
        params = {"full_samp": False, "is_cat": False, "is_date": False, "formatter": "{:.2f}"}
        result = get_num_missing(s, params)
        assert result == ["-", "-"]

    def test_get_min_max_mean_numeric(self):
        from targetviz.stats import get_min_max_mean

        s = pd.Series([1.0, 2.0, 3.0])
        params = {"full_samp": True, "is_cat": False, "is_date": False, "formatter": "{:.2f}"}
        result = get_min_max_mean(s, params)
        assert result == ["3.00", "1.00", "2.00"]

    def test_get_min_max_mean_cat(self):
        from targetviz.stats import get_min_max_mean

        s = pd.Series(["a", "b", "c"])
        params = {"full_samp": True, "is_cat": True, "is_date": False, "formatter": "{:.2f}"}
        result = get_min_max_mean(s, params)
        assert result == ["-", "-", "-"]


class TestReport:
    def test_sort_cols_by_exp_var(self):
        from targetviz.report import sort_cols_by_exp_var

        result_dict = {
            "a": {"explained_var": 0.1},
            "b": {"explained_var": 0.5},
            "c": {"explained_var": 0.3},
        }
        columns = ["a", "b", "c"]
        sorted_cols = sort_cols_by_exp_var(result_dict, columns)
        assert sorted_cols == ["b", "c", "a"]

    def test_sort_cols_by_exp_var_missing_col(self):
        from targetviz.report import sort_cols_by_exp_var

        result_dict = {
            "a": {"explained_var": 0.1},
        }
        columns = ["a", "b"]
        sorted_cols = sort_cols_by_exp_var(result_dict, columns)
        assert sorted_cols == ["a"]


class TestSmokeTest:
    """End-to-end smoke test to verify the full pipeline works."""

    def test_targetviz_report_numeric_target(self):
        from targetviz import targetviz_report

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "target": np.random.randn(n),
                "num_col": np.random.randn(n),
                "cat_col": np.random.choice(["a", "b", "c"], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test_report.html",
            )

            output_files = os.listdir(tmpdir)
            assert any("test_report" in f for f in output_files)

    def test_targetviz_report_binary_target(self):
        from targetviz import targetviz_report

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "target": np.random.choice([0, 1], n),
                "num_col": np.random.randn(n),
                "cat_col": np.random.choice(["x", "y"], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            targetviz_report(
                df,
                target="target",
                output_dir=tmpdir + os.sep,
                name_file_out="test_binary.html",
            )

            output_files = os.listdir(tmpdir)
            assert any("test_binary" in f for f in output_files)
