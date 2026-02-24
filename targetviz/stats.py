"""Descriptive statistics functions for targetviz."""

from typing import List, Union

import numpy as np
import pandas as pd

from targetviz.typedefs import DescParams


def get_num_values(series: pd.Series) -> int:
    """Get number of values in series"""
    return len(series)


def get_num_unique_values(series: pd.Series) -> int:
    """Get number of unique values in series"""
    return series.nunique()


def get_num_missing(series: pd.Series, desc_params: DescParams) -> List[Union[int, str]]:
    """Get number of missing values and percentage of total"""
    if desc_params["full_samp"]:
        n_nan = np.sum(series.isna())
        list_missing = [n_nan, "{:.2f}%".format(n_nan / len(series))]
    else:
        list_missing = ["-"] * 2
    return list_missing


def get_min_max_mean(series: pd.Series, desc_params: DescParams) -> List[Union[float, str]]:
    """Get min max and mean values of series"""
    formatter = desc_params["formatter"]
    if desc_params["is_cat"]:
        list_min_max_mean = ["-"] * 3
    else:
        list_min_max_mean = [
            formatter.format(series.max()),
            formatter.format(series.min()),
            formatter.format(series.mean()),
        ]
    return list_min_max_mean


def get_median(series: pd.Series, desc_params: DescParams) -> Union[float, str]:
    """Get median value from series"""
    formatter = desc_params["formatter"]
    if desc_params["is_cat"]:
        median = "-"
    elif desc_params["is_date"]:
        median = formatter.format(series.quantile(0.5))
    else:
        median = formatter.format(series.median())
    return median


def get_mode(series: pd.Series, desc_params: DescParams) -> Union[float, str]:
    """Get mode from series"""
    formatter = desc_params["formatter"]
    if desc_params["is_cat"]:
        mode = series.mode()[0]
    else:
        mode = formatter.format(series.mode()[0])
    return mode


def get_std(series: pd.Series, desc_params: DescParams) -> Union[float, str]:
    """Get standard deviation from series"""
    formatter = desc_params["formatter"]
    if desc_params["is_cat"]:
        std = "-"
    elif desc_params["is_date"]:
        std = formatter.format(series.sub(pd.Timestamp("2010-01-01")).dt.days.std())
    else:
        std = formatter.format(series.std())
    return std


def get_quantiles(
    series: pd.Series, desc_params: DescParams, quantiles: List[float]
) -> List[Union[float, str]]:
    """Get quantile values from series"""
    formatter = desc_params["formatter"]
    if desc_params["is_cat"]:
        list_quantiles = ["-"] * len(quantiles)
    else:
        list_quantiles = [formatter.format(series.quantile(quantile)) for quantile in quantiles]
    return list_quantiles
