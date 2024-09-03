"""Configuration for the package is handled in this wrapper for confuse.
https://github.com/pandas-profiling/pandas-profiling/blob/a55ab613865cbd0a11f2056ca6f6ca1c43c0a2f1/pandas_profiling/config.py
"""

import argparse
from pathlib import Path

import confuse


def get_config_default() -> Path():
    """Returns the path to the default config_ file.
    Returns:
        The path to the default config_ file.
    """
    return Path(__file__).parent / "config_default.yaml"


class Config(object):
    """This is a wrapper for the python confuse package, which handles setting and
    getting configuration variables via various ways (notably via argparse and kwargs).
    """

    config = None
    """The confuse.Configuration object."""

    def __init__(self):
        """The config_ constructor should be called only once."""
        if self.config is None:
            self.config = confuse.Configuration("PandasProfiling", __name__)
            self.config.set_file(str(get_config_default()))

    def set_args(self, namespace: argparse.Namespace, dots: bool) -> None:
        """
        Set config_ variables based on the argparse Namespace object.
        Args:
            namespace: Dictionary or Namespace to overlay this config_ with. Supports
                nested Dictionaries and Namespaces.
            dots: If True, any properties on namespace that contain dots (.) will be
                broken down into child dictionaries.
        """
        self.config.set_args(namespace, dots)

    def _set_kwargs(self, reference, values: dict):
        """Helper function to set config_ variables based on kwargs."""
        for key, value in values.items():
            if key in reference:
                if isinstance(value, dict):
                    self._set_kwargs(reference[key], value)
                else:
                    reference[key].set(value)
            else:
                raise ValueError('Config parameter "{}" does not exist.'.format(key))

    def set_kwargs(self, kwargs) -> None:
        """
        Helper function to set config_ variables based on kwargs.
        Args:
            kwargs: the arguments passed to the .profile_report() function
        """
        self._set_kwargs(self.config, kwargs)

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key].set(value)


config = Config()
