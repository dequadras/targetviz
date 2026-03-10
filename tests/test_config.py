"""Tests for targetviz.config module."""

import pytest

from targetviz.config import Settings, _get_config_default, _SubConfig


class TestGetConfigDefault:
    def test_returns_path_to_yaml(self):
        path = _get_config_default()
        assert path.exists()
        assert path.name == "config_default.yaml"

    def test_yaml_is_readable(self):
        import yaml

        path = _get_config_default()
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)
        assert "pct_outliers" in data


class TestSubConfig:
    def test_init_sets_attrs(self):
        sc = _SubConfig({"a": 1, "b": "hello"})
        assert sc.a == 1
        assert sc.b == "hello"

    def test_init_empty(self):
        sc = _SubConfig({})
        assert sc.__dict__ == {}

    def test_repr(self):
        sc = _SubConfig({"x": 42})
        r = repr(sc)
        assert "_SubConfig(" in r
        assert "x=42" in r

    def test_repr_multiple_attrs(self):
        sc = _SubConfig({"a": 1, "b": "two"})
        r = repr(sc)
        assert "a=1" in r
        assert "b='two'" in r


class TestSettings:
    def test_init_loads_defaults(self):
        s = Settings()
        # Check a few known defaults from config_default.yaml
        assert hasattr(s, "pct_outliers")
        assert hasattr(s, "n_breaks")
        assert hasattr(s, "quantiles")
        assert isinstance(s.quantiles, list)

    def test_nested_keys_are_subconfig(self):
        s = Settings()
        assert isinstance(s.hist, _SubConfig)
        assert isinstance(s.kde, _SubConfig)
        assert isinstance(s.heatmap, _SubConfig)

    def test_nested_attrs(self):
        s = Settings()
        assert hasattr(s.hist, "max_values")
        assert hasattr(s.kde, "max_sample")
        assert hasattr(s.kde, "ind")
        assert hasattr(s.heatmap, "cmap")

    def test_runtime_values_initialized(self):
        s = Settings()
        assert s.timestamp == ""
        assert s.target_type == ""


class TestSettingsSetKwargs:
    def test_set_top_level_value(self):
        s = Settings()
        s.set_kwargs({"pct_outliers": 0.10})
        assert s.pct_outliers == 0.10

    def test_set_multiple_values(self):
        s = Settings()
        s.set_kwargs({"pct_outliers": 0.0, "n_breaks": 5})
        assert s.pct_outliers == 0.0
        assert s.n_breaks == 5

    def test_set_nested_value(self):
        s = Settings()
        s.set_kwargs({"hist": {"max_values": 50}})
        assert s.hist.max_values == 50

    def test_set_nested_preserves_other_attrs(self):
        s = Settings()
        original_ind = s.kde.ind
        s.set_kwargs({"kde": {"max_sample": 5000}})
        assert s.kde.max_sample == 5000
        assert s.kde.ind == original_ind  # untouched

    def test_invalid_top_level_key_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="does not exist"):
            s.set_kwargs({"nonexistent_param": 42})

    def test_invalid_nested_key_raises(self):
        s = Settings()
        with pytest.raises(ValueError, match="does not exist"):
            s.set_kwargs({"hist": {"fake_param": 100}})

    def test_empty_kwargs(self):
        s = Settings()
        original_pct = s.pct_outliers
        s.set_kwargs({})
        assert s.pct_outliers == original_pct

    def test_set_string_value(self):
        s = Settings()
        s.set_kwargs({"name_file_out": "custom_report.html"})
        assert s.name_file_out == "custom_report.html"

    def test_set_title(self):
        s = Settings()
        s.set_kwargs({"title": "My Custom Title"})
        assert s.title == "My Custom Title"
