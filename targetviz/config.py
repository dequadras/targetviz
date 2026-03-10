"""Configuration for the targetviz package.

Defaults are loaded from config_default.yaml. Users can override any value
via keyword arguments passed to ``targetviz_report()``.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

_NESTED_KEYS = {"hist", "kde", "heatmap"}


def _get_config_default() -> Path:
    """Return the path to the bundled default config file."""
    return Path(__file__).parent / "config_default.yaml"


class _SubConfig:
    """Lightweight namespace for nested config sections (hist, kde, heatmap)."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"_SubConfig({attrs})"


class Settings:
    """Configuration settings for targetviz.

    All defaults are loaded from ``config_default.yaml``.
    Override any setting via kwargs passed to ``targetviz_report()``.
    """

    def __init__(self) -> None:
        with open(_get_config_default(), "r", encoding="utf-8") as fh:
            defaults: Dict[str, Any] = yaml.safe_load(fh)

        for key, value in defaults.items():
            if key in _NESTED_KEYS and isinstance(value, dict):
                setattr(self, key, _SubConfig(value))
            else:
                setattr(self, key, value)

        # Runtime values (set dynamically during report generation)
        self.timestamp: str = ""
        self.target_type: str = ""

    def set_kwargs(self, kwargs: dict) -> None:
        """Apply user-supplied keyword arguments to the config.

        Supports nested dicts for sub-configs (e.g. ``hist``, ``kde``, ``heatmap``).
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Config parameter "{key}" does not exist.')
            if isinstance(value, dict):
                sub = getattr(self, key)
                for sub_key, sub_value in value.items():
                    if not hasattr(sub, sub_key):
                        raise ValueError(f'Config parameter "{key}.{sub_key}" does not exist.')
                    setattr(sub, sub_key, sub_value)
            else:
                setattr(self, key, value)


config = Settings()
