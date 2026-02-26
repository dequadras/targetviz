"""Utility functions for targetviz."""

import logging
import os
import sys
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files


def _get_template_path(fname: str) -> os.PathLike:
    """
    Returns the full path to a template file
    """
    return files("targetviz.templates").joinpath(fname)


def create_log() -> logging.Logger:
    """
    Create logger and formatters for console output.

    Uses an explicit StreamHandler on the ``targetviz`` logger so that
    messages are visible even in environments where the root logger is
    already configured (Kaggle / Databricks / Colab notebooks).
    """
    log = logging.getLogger("targetviz")

    if not log.handlers:
        log_format = "%(asctime)s %(levelname)-8s %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        log.addHandler(handler)

    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    return log


def get_df_small(data: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """
    Generate a smaller version of the dataframe with only the specific variable and the
     target variable. This dataframe can be mutated
    """
    if col == target:
        dfs = data.loc[:, [col]].copy()
    else:
        dfs = data.loc[:, [col, target]].copy()
    return dfs


def plot_360_n0sc0pe() -> str:
    """Saves the current plot directly as an SVG string.

    Returns:
        A string containing the SVG representation of the plot.
    """
    svg_buffer = StringIO()
    # Save directly to SVG string. bbox_inches="tight" helps prevent cropping.
    plt.savefig(svg_buffer, format="svg")
    plt.close()  # Close the figure to free memory
    svg_content = svg_buffer.getvalue()
    svg_buffer.close()

    return svg_content
