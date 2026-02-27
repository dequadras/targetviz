"""Analyzer classes for targetviz."""

import logging
import warnings
from datetime import date as _date_type
from datetime import datetime as _datetime_type
from typing import Any, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns

from targetviz.config import Settings
from targetviz.stats import (
    get_median,
    get_min_max_mean,
    get_mode,
    get_num_missing,
    get_num_unique_values,
    get_num_values,
    get_quantiles,
    get_std,
)
from targetviz.typedefs import ResultDict
from targetviz.utils import get_df_small, plot_360_n0sc0pe
from targetviz.visualize import plot_kde, truncate_labels


def _clean_fp_noise(x: float) -> float:
    """Remove floating-point representation noise from a number.

    Tries progressively fewer decimal places (0 → 10) and returns the first
    rounded value that is within a tiny relative tolerance of the original.
    This turns artefacts like ``185.19899999999998`` into ``185.199`` while
    preserving genuinely meaningful precision (e.g. ``1.23456`` stays as-is).
    """
    if x == 0 or not np.isfinite(x):
        return x
    for d in range(0, 11):
        r = round(x, d)
        if r == 0 and x != 0:
            continue
        if abs(r - x) <= 1e-9 * max(abs(x), 1):
            return r
    return x


def _clean_interval_categories(cut_series: pd.Series) -> pd.Series:
    """Clean floating-point noise from Interval category labels.

    After ``pd.qcut``, interval edges can carry floating-point artefacts
    (e.g. ``185.19899999999998`` instead of ``185.199``).  This function
    renames the categories with cleaned-up edges while preserving the
    actual bin assignments.
    """
    cats = cut_series.cat.categories
    if not isinstance(cats, pd.IntervalIndex):
        return cut_series
    new_intervals = pd.IntervalIndex.from_arrays(
        [_clean_fp_noise(x) for x in cats.left],
        [_clean_fp_noise(x) for x in cats.right],
        closed=cats.closed,
    )
    if new_intervals.equals(cats):
        return cut_series
    return cut_series.cat.rename_categories(new_intervals)


def _is_string_or_object_dtype(dtype) -> bool:
    """Check if dtype represents string or object data.

    Handles numpy object, pandas StringDtype (including pandas 3 default str type),
    and pyarrow string/large_string.
    """
    if ptypes.is_object_dtype(dtype):
        return True
    if isinstance(dtype, pd.StringDtype):
        return True
    if hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype):
        try:
            import pyarrow as pa

            return pa.types.is_string(dtype.pyarrow_dtype) or pa.types.is_large_string(
                dtype.pyarrow_dtype
            )
        except ImportError:
            pass
    return False


def _is_datetime_dtype(dtype) -> bool:
    """Check if dtype is datetime-like.

    Handles numpy datetime64, pandas DatetimeTZDtype, and pyarrow timestamps/dates.
    """
    if ptypes.is_datetime64_any_dtype(dtype):
        return True
    try:
        if dtype.kind == "M":
            return True
    except (AttributeError, TypeError):
        pass
    if hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype):
        try:
            import pyarrow as pa

            return pa.types.is_timestamp(dtype.pyarrow_dtype) or pa.types.is_date(
                dtype.pyarrow_dtype
            )
        except ImportError:
            pass
    return False


def _has_nonscalar_values(series: pd.Series) -> Optional[str]:
    """Check if an object-dtype series contains non-scalar values (lists, dicts, etc.).

    Samples up to 50 non-null values and checks their Python types.
    Returns a string describing the unsupported types found, or None if all
    values are scalar (str, numbers, dates, booleans, None).
    """
    if not ptypes.is_object_dtype(series.dtype):
        return None
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    _scalar_types = (
        str,
        int,
        float,
        complex,
        bool,
        np.integer,
        np.floating,
        np.bool_,
        _date_type,
        _datetime_type,
    )
    sample = non_null.head(min(50, len(non_null)))
    bad_types: set = set()
    for v in sample:
        if not isinstance(v, _scalar_types):
            bad_types.add(type(v).__name__)
    if bad_types:
        return ", ".join(sorted(bad_types))
    return None


def _is_date_object_column(series: pd.Series) -> bool:
    """Check if an object-dtype series actually contains date/datetime objects.

    Samples up to 20 non-null values and checks their Python types.
    """
    if not ptypes.is_object_dtype(series.dtype):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(min(20, len(non_null)))
    return all(isinstance(v, (_date_type, _datetime_type)) for v in sample)


def _coerce_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a date-like series to pandas datetime64.

    Handles datetime.date objects, pyarrow date/timestamp types,
    and other formats supported by pd.to_datetime.
    Always returns a numpy-backed datetime64 series.
    """
    # For pyarrow-backed types, convert through Python objects first
    # to avoid pyarrow timezone database issues and ensure numpy-backed result
    if hasattr(pd, "ArrowDtype") and isinstance(series.dtype, pd.ArrowDtype):
        try:
            return pd.to_datetime(series.astype(object))
        except (ValueError, TypeError, OverflowError):
            return series
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return series
    try:
        return pd.to_datetime(series, format="mixed")
    except (ValueError, TypeError, OverflowError):
        return series


def _is_numeric_dtype(dtype) -> bool:
    """Check if dtype is numeric (int, float, uint), excluding boolean.

    Handles numpy dtypes, pandas nullable types (Int8-64, UInt8-64, Float32/64),
    and pyarrow-backed numeric types.
    """
    if ptypes.is_bool_dtype(dtype):
        return False
    return ptypes.is_numeric_dtype(dtype)


class BaseAnalyzer:
    """
    Base for analyzer classes TargetAnalyzer and ColumnAnalyzer
    """

    def __init__(self, col: str, target: str, config_: Settings, log: logging.Logger):
        self.config: Settings = config_
        self.type: Optional[Literal["UNIQUE", "BINARY", "DATE", "CAT", "NUM"]] = None
        self.rate_non_nulls: Optional[float] = None
        self.target: str = target
        self.log: logging.Logger = log
        self.col: str = col

    def get_type(self, df_small: pd.DataFrame) -> pd.DataFrame:
        """
        This functions saves the type variable to self.type, and does small changes
        """
        n_values = df_small[self.col].nunique()
        if n_values < 2:
            self.type = "UNIQUE"
        elif n_values == 2:
            dtype = df_small[self.col].dtype
            if _is_datetime_dtype(dtype) or _is_date_object_column(df_small[self.col]):
                self.type = "DATE"
                df_small[self.col] = _coerce_to_datetime(df_small[self.col])
            else:
                self.type = "BINARY"
                df_small[self.col] = df_small[self.col].astype("category")
        else:
            dtype = df_small[self.col].dtype
            if _is_datetime_dtype(dtype):
                self.type = "DATE"
                df_small[self.col] = _coerce_to_datetime(df_small[self.col])
            elif _is_date_object_column(df_small[self.col]):
                self.log.info(f"Coercing date column {self.col} from {dtype} to datetime")
                df_small[self.col] = _coerce_to_datetime(df_small[self.col])
                self.type = "DATE"
            elif _is_string_or_object_dtype(dtype):
                self.log.info(f"Coercing column {self.col} from {dtype} to category")
                df_small[self.col] = df_small[self.col].astype("category")
                self.type = "CAT"
            elif isinstance(dtype, pd.CategoricalDtype):
                self.type = "CAT"
            elif _is_numeric_dtype(dtype):
                self.type = "NUM"
            else:
                raise TypeError(f"cannot parse type {dtype}")
        return df_small

    def create_desc_df(self, series: pd.Series, series_clean: pd.Series) -> pd.DataFrame:
        """
        Creates dataFrame with main statistic values
        :param series: pd.Series with full sample of result_dict to describe
        :param series_clean: pd.Series with no outliers or nan to describe
        :return: pd.DataFrame with table to show
        """
        quantiles = self.config.quantiles

        list_fullsam = []
        list_fullsam.extend(self.get_desc(series, full_samp=True, quantiles=quantiles))
        list_reduced = []
        list_reduced.extend(self.get_desc(series_clean, full_samp=False, quantiles=quantiles))

        stat_names = [
            "Number of values",
            "Num. distinct values",
            "Number missing",
            "Percent missing",
            "Maximum",
            "Minimum",
            "Mean",
            "Median",
            "Mode",
            "Standard deviation",
        ]

        stat_names.extend([f"Percentile {quantile * 100:.0f}%" for quantile in quantiles])

        assert len(list_fullsam) == len(stat_names)
        assert len(list_reduced) == len(stat_names)

        stat_df = pd.DataFrame(
            {
                "Measurement": stat_names,
                "Full sample": list_fullsam,
                "Outliers and missings removed": list_reduced,
            }
        )

        return stat_df

    def create_desc_and_hist(self, series: pd.Series, series_clean: pd.Series) -> Tuple[str, str]:
        """
        This function gets the descriptive variables of the variable and saves a
        histogram plot and generates an HTML table
        """
        desc = self.create_desc_df(series, series_clean)

        # Create figure for histogram only
        fig, ax = plt.subplots(figsize=tuple(self.config.figures_size))

        self.plot_histogram(series_clean, ax)

        histogram = plot_360_n0sc0pe()

        # Generate HTML table
        html_table = desc.to_html(classes="table table-striped", index=False)

        return histogram, html_table

    def get_desc(self, series: pd.Series, full_samp: bool, quantiles: List[float]) -> List[Any]:
        """
        Get descriptive values both for full sample result_dict or no outliers and nan
        :param series: pd.Series with result_dict to describe
        :param full_samp: boolean, True if the full sample is there
        :param quantiles: numeric list, values from 0 to 1. Quantiles for descriptive
        analysis
        :return: list with statistics
        """
        is_cat = self.type in ["CAT", "BINARY"]
        is_date = self.type == "DATE"

        list_stat = list()

        desc_params = {
            "full_samp": full_samp,
            "is_cat": is_cat,
            "is_date": is_date,
            "formatter": "{}" if is_date else "{:.2f}",
        }

        # Replace infinite values with NaN so stats functions produce meaningful
        # results instead of triggering numpy RuntimeWarnings.
        if not is_cat and not is_date:
            series = series.replace([np.inf, -np.inf], np.nan)

        list_stat.append(get_num_values(series))
        list_stat.append(get_num_unique_values(series))
        list_stat.extend(get_num_missing(series, desc_params))
        list_stat.extend(get_min_max_mean(series, desc_params))
        list_stat.append(get_median(series, desc_params))
        list_stat.append(get_mode(series, desc_params))
        list_stat.append(get_std(series, desc_params))
        list_stat.extend(get_quantiles(series, desc_params, quantiles=quantiles))

        return list_stat

    def plot_histogram(self, series: pd.Series, ax: plt.Axes) -> None:
        """
        Plot histogram of the column, with differences across types
        """
        if self.type in ["CAT", "BINARY"]:
            series.value_counts().nlargest(self.config.hist.max_values).plot(kind="bar", ax=ax)
        elif self.type == "DATE":
            from pandas.plotting import register_matplotlib_converters

            register_matplotlib_converters()

            series.hist(ax=ax)
        else:
            ax.hist(series.to_numpy(), bins=30, density=True)
            plot_kde(series, ax, self.config)
            ax.set(xlim=(series.min(), series.max()))
            ax.axvline(x=series.mean(), color="orange", linestyle="--")
        ax.set_title(f"Distribution of {self.col}", fontsize=20)
        truncate_labels(ax, self.config)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()


class TargetAnalyzer(BaseAnalyzer):
    """
    Class for analyzing target variable
    """

    def __init__(self, target: str, config_: Settings, log: logging.Logger):
        super().__init__(target, target, config_, log)

    def check_types(self, dfs: pd.DataFrame) -> None:
        """
        Function checks that the target type is acceptable
        """
        if self.type not in ["BINARY", "CAT", "NUM"]:
            raise TypeError(f"{self.type} is not an allowed type for the target")
        if self.type == "CAT":
            assert (
                dfs[self.target].nunique() < self.config.max_target_class
            ), f"Number of categories {dfs[self.target].nunique()} is too large"

    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
        """
        method for running the main function from TargetAnalyzer
        """
        data = data.copy()  # avoid SettingWithCopyWarning when modifying target column
        dfs = get_df_small(data, self.target, self.target)
        dfs = self.get_type(dfs)
        data[self.target] = dfs[self.target]

        self.check_types(dfs)
        target_histogram, target_table = self.create_desc_and_hist(
            data[self.target], dfs[self.target]
        )

        return data, target_histogram, target_table


class ColumnAnalyzer(BaseAnalyzer):
    """
    This object does the analysis of one column of the dataframe
    """

    def clean_data(self, df_small: pd.DataFrame) -> pd.DataFrame:
        """
        Return a dataset with no nulls, infinite and remove outliers when necessary
        """
        col_series = df_small[self.col]
        mask = col_series.notna()
        if self.type == "NUM":
            mask = mask & np.isfinite(col_series.to_numpy(dtype=float, na_value=np.nan))
        df_clean = df_small[mask]
        if self.type == "NUM" and self.config.pct_outliers > 0:
            df_clean = self.remove_outliers(df_clean)
        self.rate_non_nulls = df_clean.shape[0] / df_small.shape[0]
        return df_clean

    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method removes outliers according to parameter self.pct outliers
        Only applicable for numeric data
        percentile
        """
        pct_outliers = self.config.pct_outliers
        limits = [pct_outliers / 2, 1 - pct_outliers / 2]
        col_values = data[self.col].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(col_values, [limits[0] * 100, limits[1] * 100])
        mask = (col_values >= lo) & (col_values <= hi)
        return data[mask]

    def sanity_checks(self, data: pd.DataFrame) -> bool:
        """
        Do some sanity checks necessary and return a boolean on weather the tests are
        passed
        """
        if data[self.col].nunique() == 0:
            return False
        return True

    def change_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Do some type changing before starting, object and string types are not allowed.
        Date objects in object columns are left untouched for later detection.
        """
        if _is_string_or_object_dtype(data[self.col].dtype):
            if not _is_date_object_column(data[self.col]):
                data[self.col] = data[self.col].astype("category")
        return data

    def run(self, data: pd.DataFrame, result_dict: ResultDict) -> ResultDict:
        """
        Main function for running the column analysis
        """
        # Check for non-scalar values (lists, dicts, etc.) in object columns
        # before any other check, since unhashable types crash nunique()/etc.
        bad_types = _has_nonscalar_values(data[self.col])
        if bad_types:
            result_dict.setdefault("skipped_variables", []).append(
                {"name": self.col, "reason": f"Contains non-scalar values ({bad_types})"}
            )
            self.log.warning(f"Skipping column {self.col}: contains non-scalar types: {bad_types}")
            return result_dict

        if not self.sanity_checks(data):
            result_dict.setdefault("skipped_variables", []).append(
                {"name": self.col, "reason": "All values are missing (no non-null values)"}
            )
            self.log.warning(f"Skipping column {self.col}: all values are missing")
            return result_dict
        df_small = get_df_small(data, self.col, self.target)
        df_small = self.change_types(df_small)
        df_small = self.get_type(df_small)
        if self.type == "UNIQUE":
            result_dict.setdefault("skipped_variables", []).append(
                {"name": self.col, "reason": "Constant (only one unique value)"}
            )
            self.log.warning(f"Skipping column {self.col} with only one value")
            return result_dict

        df_clean = self.clean_data(df_small)
        # Create plot of histogram, KDE and descriptive statistics
        col_histogram, col_table = self.create_desc_and_hist(df_small[self.col], df_clean[self.col])
        result_dict[self.col] = {"histogram": col_histogram, "table": col_table}

        cut_series = self.get_buckets(df_clean)

        # Calc explained variance
        explained_var = self.calc_explained_variance(df_clean, cut_series)
        result_dict[self.col]["explained_var"] = explained_var

        fig = plt.figure(figsize=tuple(self.config.figures_size))
        fig.suptitle("Relation between column {} and target".format(self.col), size=16)

        # Scatter for continuous and boxplot for category
        self.plot_var_target_relation(df_clean, cut_series)

        # Relation plot
        self.plot_cut_var_target_relation(df_clean, cut_series)

        rel_fig = plot_360_n0sc0pe()
        result_dict[self.col]["rel_fig"] = rel_fig

        # Compute expected target values for null/not-null and outlier/not-outlier
        target_stats_html = self.compute_target_expected_values(df_small)
        result_dict[self.col]["target_stats"] = target_stats_html

        return result_dict

    def plot_cut_var_target_relation(self, df_small: pd.DataFrame, cut_col: pd.Series) -> None:
        """
        Plots of relation between buckets of column and target value
        """
        ax1 = plt.subplot2grid((2, 2), (0, 1))
        # add y label to ax1
        target_type = self.config.target_type
        if target_type == "NUM":
            df_small[self.target].groupby(cut_col, observed=False).mean().plot(ax=ax1)
            truncate_labels(ax1, self.config)
            plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        elif target_type in ["BINARY", "CAT"]:
            target_values = np.sort(df_small[self.target].unique())
            legend_vals = []
            for target_val in target_values:
                if (target_val == 0) & (target_type == "BINARY"):
                    # for binary plot plot only one class
                    if not self.config.plot_0_in_binary_target:
                        continue
                (df_small[self.target] == target_val).groupby(cut_col, observed=False).mean().plot(
                    ax=ax1, marker="o"
                )
                legend_vals.append(target_val)
            plt.legend(legend_vals)
            truncate_labels(ax1, self.config)
            plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        ax1.set_ylabel(self.target + " mean")
        ax2 = plt.subplot2grid((2, 2), (1, 1), sharex=ax1)
        cut_col.value_counts().sort_index().plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Count")
        truncate_labels(ax2, self.config)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

    def plot_var_target_relation(self, df_small: pd.DataFrame, cut_col: pd.Series) -> None:
        """
        Plot between column and target variable (scatter for numeric to numeric)
        """
        ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        target_type: str = self.config.target_type
        cmap: str = self.config.heatmap.cmap
        max_scatter_points: int = self.config.max_scatter_points

        if self.type in ["CAT", "BINARY", "DATE"]:
            if target_type == "NUM":
                with warnings.catch_warnings():
                    # seaborn internally passes vert= to matplotlib; suppress until seaborn updates
                    warnings.filterwarnings(
                        "ignore", message="vert", category=PendingDeprecationWarning
                    )
                    # seaborn <0.13 calls groupby without observed=; suppress pandas FutureWarning
                    warnings.filterwarnings("ignore", message="observed", category=FutureWarning)
                    sns.boxplot(x=cut_col, y=df_small[self.target], showfliers=False, ax=ax0)
                truncate_labels(ax0, self.config)
                ax0.set_xticks(range(len(ax0.get_xticklabels())))
                ax0.set_xticklabels(ax0.get_xticklabels(), rotation=30, ha="right")
            elif target_type in ["BINARY", "CAT"]:
                data: pd.DataFrame = pd.DataFrame(
                    data={self.col: cut_col, self.target: df_small[self.target]}
                )
                data = (
                    data.groupby([self.col, self.target], observed=False).size() / data.shape[0]
                ).reset_index()
                data = data.pivot(index=self.col, columns=self.target, values=0)
                data = data / data.sum()  # normalize
                sns.heatmap(data, cmap=cmap, ax=ax0)
                truncate_labels(ax0, self.config)
        else:
            if target_type == "NUM":
                if df_small.shape[0] > max_scatter_points:
                    df_sample: pd.DataFrame = df_small.sample(max_scatter_points)
                    sns.scatterplot(x=df_sample[self.col], y=df_sample[self.target], ax=ax0)
                else:
                    sns.scatterplot(x=df_small[self.col], y=df_small[self.target], ax=ax0)
                ax0.set(xlim=(df_small[self.col].min(), df_small[self.col].max()))
                truncate_labels(ax0, self.config)
            elif target_type in ["BINARY", "CAT"]:
                with warnings.catch_warnings():
                    # seaborn internally passes vert= to matplotlib; suppress until seaborn updates
                    warnings.filterwarnings(
                        "ignore", message="vert", category=PendingDeprecationWarning
                    )
                    # seaborn <0.13 calls groupby without observed=; suppress pandas FutureWarning
                    warnings.filterwarnings("ignore", message="observed", category=FutureWarning)
                    sns.boxplot(
                        x=df_small[self.col],
                        y=df_small[self.target],
                        ax=ax0,
                        showfliers=False,
                    )
                truncate_labels(ax0, self.config)
        plt.title(" ", fontsize=20)  # necessary so that space is left for main title

    def get_buckets(self, df_small: pd.DataFrame) -> pd.Series:
        """
        Function to cut column into different buckets, for later analysis
        """
        n_breaks = self.config.n_breaks
        if self.type in ["CAT", "BINARY"]:
            val_counts = df_small[self.col].value_counts().sort_values(ascending=False)
            if len(val_counts) > n_breaks:
                top_cat = val_counts.nlargest(n_breaks - 1).index.values.astype("str")
                other = "Other"
                cut_col = df_small[self.col].copy()
                if other not in cut_col.cat.categories:
                    cut_col = cut_col.cat.add_categories(other)
                else:
                    self.log.warning(
                        f"Value '{other}' is  already in categories, which can cause "
                        f"problems with the buckets"
                    )
                cut_col[~cut_col.isin(top_cat)] = other
                cut_col = cut_col.cat.remove_unused_categories()
            else:
                cut_col = df_small[self.col]
        else:
            rows_before = df_small.shape[0]
            assert df_small[self.col].isna().sum() == 0
            pct_rows_removed = (rows_before - df_small.shape[0]) / rows_before * 100
            self.log.info(f"removed {pct_rows_removed:.0f}% nan")

            if df_small[self.col].nunique() <= 2:
                # variable only takes two values
                cut_col = df_small[self.col].astype("category")
            else:
                if self.type == "DATE":
                    cut_col = pd.qcut(
                        df_small[self.col].astype(np.int64),
                        n_breaks,
                        duplicates="drop",
                        precision=self.config.qcut_precision,
                    )
                    if len(cut_col.cat.categories) != cut_col.nunique():
                        cut_col = cut_col.cat.remove_unused_categories()
                    cut_col = _clean_interval_categories(cut_col)

                else:
                    cut_col = pd.qcut(
                        df_small[self.col],
                        n_breaks,
                        duplicates="drop",
                        precision=self.config.qcut_precision,
                    )
                    if len(cut_col.cat.categories) != cut_col.nunique():
                        cut_col = cut_col.cat.remove_unused_categories()
                    cut_col = _clean_interval_categories(cut_col)
        if cut_col.nunique() == 1:
            warning_msg = (
                f"Column {self.col} has only one bucket. This can be caused\n"
                "by a numeric variable being dominated by one number"
            )
            self.log.warning(warning_msg)
        return cut_col

    def _format_expected_value(self, target_series: pd.Series) -> str:
        """Format expected value of target series based on target type."""
        target_type = self.config.target_type
        if target_type == "NUM":
            return f"{target_series.mean():.4f}"
        else:
            # BINARY or CAT
            vc = target_series.value_counts(normalize=True).sort_index()
            parts = [f"P({v})={r:.2%}" for v, r in vc.items()]
            return ", ".join(parts)

    @staticmethod
    def _format_expected_value_np(
        target_arr: np.ndarray, target_type: str, unique_classes: Optional[np.ndarray] = None
    ) -> str:
        """Format expected value from a numpy array (fast path)."""
        if target_type == "NUM":
            return f"{np.mean(target_arr):.4f}"
        else:
            # BINARY or CAT — compute normalised counts via numpy
            if unique_classes is None:
                unique_classes = np.unique(target_arr)
            n = len(target_arr)
            parts = []
            for cls in unique_classes:
                rate = np.sum(target_arr == cls) / n
                parts.append(f"P({cls})={rate:.2%}")
            return ", ".join(parts)

    def compute_target_expected_values(self, df_small: pd.DataFrame) -> str:
        """Compute expected value of target for null/not-null and outlier/not-outlier groups.

        Returns an HTML string with the stats.
        """
        # Extract numpy arrays once — avoids repeated pandas indexing overhead
        col_arr = df_small[self.col].to_numpy()
        target_arr = df_small[self.target].to_numpy()
        target_type = self.config.target_type

        # Pre-compute sorted unique classes for BINARY/CAT (reused in every call)
        unique_classes = np.unique(target_arr) if target_type != "NUM" else None

        stats_parts: list[str] = []

        # --- Null vs not-null ---
        null_mask = pd.isna(col_arr)
        null_count = int(null_mask.sum())
        not_null_count = int((~null_mask).sum())

        if null_count > 0 and not_null_count > 0:
            ev_null = self._format_expected_value_np(
                target_arr[null_mask], target_type, unique_classes
            )
            ev_not_null = self._format_expected_value_np(
                target_arr[~null_mask], target_type, unique_classes
            )
            stats_parts.append(
                f"<b>E[{self.target} | {self.col} is null]</b> = {ev_null} &nbsp;(n={null_count})"
            )
            stats_parts.append(
                f"<b>E[{self.target} | {self.col} is not null]</b> = {ev_not_null} "
                f"&nbsp;(n={not_null_count})"
            )
        elif null_count == 0:
            stats_parts.append(f"<i>No null values in {self.col}</i>")

        # --- Outlier vs not-outlier (numeric predictors only) ---
        if self.type == "NUM" and self.config.pct_outliers > 0:
            pct_outliers = self.config.pct_outliers
            pct_lo = pct_outliers / 2 * 100  # convert to percentile for numpy
            pct_hi = (1 - pct_outliers / 2) * 100

            # Work on non-null, finite values only — pure numpy, no DataFrame copy
            col_float = col_arr.astype(float, copy=False)
            finite_mask = (~null_mask) & np.isfinite(col_float)
            col_finite = col_float[finite_mask]
            target_finite = target_arr[finite_mask]

            if col_finite.shape[0] > 0:
                # Single numpy percentile call for both bounds
                lower, upper = np.nanpercentile(col_finite, [pct_lo, pct_hi])
                outlier_mask = (col_finite < lower) | (col_finite > upper)
                outlier_count = int(outlier_mask.sum())
                not_outlier_count = int((~outlier_mask).sum())

                if outlier_count > 0 and not_outlier_count > 0:
                    ev_outlier = self._format_expected_value_np(
                        target_finite[outlier_mask], target_type, unique_classes
                    )
                    ev_not_outlier = self._format_expected_value_np(
                        target_finite[~outlier_mask], target_type, unique_classes
                    )
                    stats_parts.append(
                        f"<b>E[{self.target} | {self.col} is outlier]</b> = {ev_outlier} "
                        f"&nbsp;(n={outlier_count})"
                    )
                    stats_parts.append(
                        f"<b>E[{self.target} | {self.col} is not outlier]</b> = {ev_not_outlier} "
                        f"&nbsp;(n={not_outlier_count})"
                    )
                else:
                    stats_parts.append(f"<i>No outliers detected in {self.col}</i>")

        return "<br>".join(stats_parts)

    def calc_explained_variance(self, data: pd.DataFrame, cut_series: pd.Series) -> float:
        """
        Calculate explained variance when binning into n groups.
        It is a proxy of the importance of that variable for a future model.
        The variance is weighted by the rate of non nulls
        """
        assert cut_series.isna().sum() == 0
        assert data.shape[0] == cut_series.shape[0]
        target_type = self.config.target_type
        series_target = data[self.target]
        if target_type == "NUM":
            return self.calc_explained_variance_num(series_target, cut_series)
        elif target_type in ["BINARY", "CAT"]:
            return self.calc_explained_variance_cat(series_target, cut_series)

    def calc_explained_variance_num(self, series_target: pd.Series, cut_series: pd.Series) -> float:
        """
        Perform calc_explained_variance for the case of numeric target
        """
        explained_variance = self.calc_explained_variance_(series_target, cut_series)

        return explained_variance * self.rate_non_nulls

    def calc_explained_variance_cat(self, series_target: pd.Series, cut_series: pd.Series) -> float:
        """
        Perform calc_explained_variance for the case of categorical/binary target.
        """
        exp_var_list = []
        for target_val in series_target.unique():
            series_target_cat = series_target == target_val
            exp_var_list.append(self.calc_explained_variance_(series_target_cat, cut_series))
        return np.mean(exp_var_list) * self.rate_non_nulls

    @staticmethod
    def calc_explained_variance_(series_target: pd.Series, cut_series: pd.Series) -> float:
        """
        Generic method for calc explained variance.

        Uses pure numpy (np.bincount) instead of pandas groupby for speed.
        """
        target = np.asarray(series_target, dtype=np.float64)
        n = target.shape[0]

        total_ss = target.var(ddof=0) * n  # = sum of (x - mean)^2
        if total_ss == 0:
            return 0.0

        # Obtain integer group codes from the cut_series
        if hasattr(cut_series, "cat"):
            codes = cut_series.cat.codes.values
        elif hasattr(cut_series, "codes"):
            codes = cut_series.codes
        else:
            _, codes = np.unique(np.asarray(cut_series), return_inverse=True)

        n_groups = int(codes.max()) + 1

        # Per-group sums and counts via bincount (single pass each)
        group_sums = np.bincount(codes, weights=target, minlength=n_groups)
        group_counts = np.bincount(codes, minlength=n_groups).astype(np.float64)

        # Per-group means (safe division)
        safe_counts = np.where(group_counts > 0, group_counts, 1.0)
        group_means = group_sums / safe_counts

        # Within-group SS = sum of (x_i - group_mean_i)^2
        residuals = target - group_means[codes]
        within_ss = residuals.dot(residuals)

        return 1.0 - within_ss / total_ss
