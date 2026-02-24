"""Plotting utility functions for targetviz."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from targetviz.config import Settings


def plot_kde(series: pd.Series, ax: plt.Axes, config_: Settings) -> None:
    """
    Plot kde with config_ parameters
    """
    max_sample = config_.kde.max_sample
    ind = config_.kde.ind
    if len(series) > max_sample:
        series = series.sample(max_sample)
    try:
        series.plot.kde(ind=ind, ax=ax)
    except np.linalg.LinAlgError:
        print("not able to kde")


def truncate_labels(ax, config: Settings):
    """
    Truncate labels on the given axis to ensure they have at most a certain length.
    """
    max_lable_len = config.max_lable_len
    # Truncate x-axis labels
    xlabels = ax.get_xticks()
    new_xlabels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        if len(text) > max_lable_len:
            truncated_text = text[: max_lable_len - 3] + "..."
        else:
            truncated_text = text
        new_xlabels.append(truncated_text)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(new_xlabels)

    # Truncate y-axis labels
    ylabels = ax.get_yticks()
    new_ylabels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        if len(text) > max_lable_len:
            truncated_text = text[: max_lable_len - 3] + "..."
        else:
            truncated_text = text
        new_ylabels.append(truncated_text)
    ax.set_yticks(ylabels)
    ax.set_yticklabels(new_ylabels)
