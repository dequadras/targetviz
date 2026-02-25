"""Tests for targetviz.report module — set_default_params and render_output."""

import os
import tempfile
import zipfile

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from targetviz.config import Settings  # noqa: E402
from targetviz.report import (  # noqa: E402
    render_output,
    set_default_params,
    sort_cols_by_exp_var,
)

# ===================================================================
# sort_cols_by_exp_var — edge cases
# ===================================================================


class TestSortColsByExpVar:
    def test_empty_result_dict(self):
        result = sort_cols_by_exp_var({}, ["a", "b", "c"])
        assert result == []

    def test_all_columns_missing(self):
        result_dict = {"x": {"explained_var": 0.5}}
        result = sort_cols_by_exp_var(result_dict, ["a", "b"])
        assert result == []

    def test_single_column(self):
        result_dict = {"a": {"explained_var": 0.5}}
        result = sort_cols_by_exp_var(result_dict, ["a"])
        assert result == ["a"]

    def test_equal_variance(self):
        result_dict = {
            "a": {"explained_var": 0.5},
            "b": {"explained_var": 0.5},
        }
        result = sort_cols_by_exp_var(result_dict, ["a", "b"])
        assert set(result) == {"a", "b"}
        assert len(result) == 2


# ===================================================================
# set_default_params
# ===================================================================


class TestSetDefaultParams:
    def test_default_name(self):
        config_ = Settings()
        config_.timestamp = "2026_01_01__12_00_00"
        config_.name_file_out = "default"
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        columns, name = set_default_params(config_, None, "target", df)
        assert "2026_01_01__12_00_00" in name
        assert name.endswith(".html")

    def test_custom_name_html(self):
        config_ = Settings()
        config_.name_file_out = "my_report.html"
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        columns, name = set_default_params(config_, ["col1"], "target", df)
        assert name == "my_report.html"

    def test_custom_name_zip(self):
        config_ = Settings()
        config_.name_file_out = "my_report.html.zip"
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        columns, name = set_default_params(config_, ["col1"], "target", df)
        assert name == "my_report.html.zip"

    def test_name_without_extension_gets_html(self):
        config_ = Settings()
        config_.name_file_out = "my_report"
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        columns, name = set_default_params(config_, ["col1"], "target", df)
        assert name == "my_report.html"

    def test_columns_none_uses_all_except_target(self):
        config_ = Settings()
        config_.name_file_out = "test.html"
        df = pd.DataFrame({"target": [0, 1], "a": [1, 2], "b": [3, 4]})
        columns, _ = set_default_params(config_, None, "target", df)
        assert set(columns) == {"a", "b"}

    def test_columns_explicit(self):
        config_ = Settings()
        config_.name_file_out = "test.html"
        df = pd.DataFrame({"target": [0, 1], "a": [1, 2], "b": [3, 4]})
        columns, _ = set_default_params(config_, ["a"], "target", df)
        assert columns == ["a"]

    def test_columns_as_tuple(self):
        config_ = Settings()
        config_.name_file_out = "test.html"
        df = pd.DataFrame({"target": [0, 1], "a": [1, 2], "b": [3, 4]})
        columns, _ = set_default_params(config_, ("a", "b"), "target", df)
        assert columns == ("a", "b")


# ===================================================================
# render_output — integration
# ===================================================================


class TestRenderOutput:
    def _make_result_dict(self):
        """Build a minimal result_dict for rendering."""
        return {
            "target": "target",
            "target_histogram": "<svg>target_hist</svg>",
            "target_table": "<table>target_table</table>",
            "col1": {
                "histogram": "<svg>col1_hist</svg>",
                "table": "<table>col1_table</table>",
                "explained_var": 0.42,
                "rel_fig": "<svg>col1_rel</svg>",
            },
        }

    def test_creates_html_file(self):
        result_dict = self._make_result_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.html")
            render_output(result_dict, ["col1"], path)
            assert os.path.exists(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<html" in content.lower() or "<body" in content.lower() or "target" in content

    def test_creates_zip_file(self):
        result_dict = self._make_result_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.html.zip")
            render_output(result_dict, ["col1"], path)
            assert os.path.exists(path)
            assert zipfile.is_zipfile(path)
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()
                assert any(n.endswith(".html") for n in names)
