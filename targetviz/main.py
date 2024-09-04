"""
Module containing all functionality for the targetviz report
"""

import base64
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template

from targetviz.config import config

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files

# Define more specific types
ConfigDict = Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]
ResultDict = Dict[str, Dict[str, Union[str, float, Dict[str, Any]]]]


class DescParams(TypedDict):
    full_samp: bool
    is_cat: bool
    is_date: bool
    formatter: Callable[[Union[float, datetime]], str]


def _get_template_path(fname: str) -> os.PathLike:
    """
    Returns the full path to a template file
    """
    return files("targetviz.templates").joinpath(fname)


def create_log(timestamp: str, output_dir: str, log_name: Optional[str] = None) -> logging.Logger:
    """
    Create logger and formatters, and handlers
    """
    if log_name is None:
        log_name = f"targetviz_analysis_{timestamp}.log"
    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        format=log_format, level=os.environ.get("LOGLEVEL", "INFO"), datefmt=date_format
    )

    log = logging.getLogger("targetviz")

    file_handler = logging.FileHandler(os.path.join(output_dir, log_name))

    formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

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


def sort_cols_by_exp_var(result_dict: ResultDict, columns: List[str]) -> List[str]:
    """
    get columns by order according to explained variance
    remove columns that are not analyzed
    """
    expl_var_list = []
    used_cols = []
    for col in columns:
        if col in result_dict.keys():
            expl_var_list.append(result_dict[col]["explained_var"])
            used_cols.append(col)

    order = [x for _, x in sorted(zip(expl_var_list, range(len(expl_var_list))))]
    used_cols = [used_cols[ord_] for ord_ in reversed(order)]
    return used_cols


def plot_kde(series: pd.Series, ax: plt.Axes, config_: ConfigDict) -> None:
    """
    Plot kde with config_ parameters
    """
    max_sample = config_["kde"]["max_sample"].get(int)
    ind = config_["kde"]["ind"].get(int)
    if len(series) > max_sample:
        series = series.sample(max_sample)
    try:
        series.plot.kde(ind=ind, ax=ax)
    except np.linalg.LinAlgError:
        print("not able to kde")


def render_output(result_dict: ResultDict, columns: List[str], name_html: str) -> None:
    """
    Create html file, populate and render file
    """

    # Create html file and render
    base_html = _get_template_path("base.html")

    # fill jinja template with data
    with open(base_html, "r") as file:
        template = Template(file.read())
    extra_html = _get_template_path("extra_col.html")

    used_cols = sort_cols_by_exp_var(result_dict, columns)
    html = template.render(result_dict=result_dict, columns=columns)

    with open(extra_html, "r") as file:
        extra_template = Template(file.read())

    for col in used_cols:
        html_out = extra_template.render(result_dict=result_dict, column=col)
        html += html_out

    # write to file
    with open(name_html, "w") as f:
        f.write(html)


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


def truncate_labels(ax, config: ConfigDict):
    """
    Truncate labels on the given axis to ensure they have at most a certain length.
    """
    max_lable_len = config["max_lable_len"].get(int)
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


class BaseAnalyzer:
    """
    Base for analyzer classes TargetAnalyzer and ColumnAnalyzer
    """

    def __init__(self, col: str, target: str, config_: ConfigDict, log: logging.Logger):
        self.config: ConfigDict = config_
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
            self.type = "BINARY"
            df_small[self.col] = df_small[self.col].astype("category")
        else:
            type_ = df_small[self.col].dtype.name
            if type_ == "object":
                self.log.info(f"Coercing column {self.col} from object to category")
                df_small[self.col] = df_small[self.col].astype("category")

            if type_ in ["datetime64[ns]"]:
                self.type = "DATE"
            elif type_ in ["category", "object"]:
                self.type = "CAT"
            elif type_.startswith(("float", "int", "Int", "Float")):
                self.type = "NUM"
            else:
                raise TypeError(f"cannot parse type {type_}")
        return df_small

    def create_desc_df(self, series: pd.Series, series_clean: pd.Series) -> pd.DataFrame:
        """
        Creates dataFrame with main statistic values
        :param series: pd.Series with full sample of result_dict to describe
        :param series_clean: pd.Series with no outliers or nan to describe
        :return: pd.DataFrame with table to show
        """
        quantiles = self.config["quantiles"].get(list)

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
        fig, ax = plt.subplots(figsize=tuple(self.config["figures_size"].get(list)))

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
            series.value_counts().nlargest(self.config["hist"]["max_values"].get(int)).plot(
                kind="bar", ax=ax
            )
        elif self.type == "DATE":
            from pandas.plotting import register_matplotlib_converters

            register_matplotlib_converters()

            series.hist(ax=ax)
        else:
            series.hist(bins=30, density=1, ax=ax)
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

    def __init__(self, target: str, config_: ConfigDict, log: logging.Logger):
        super().__init__(target, target, config_, log)

    def check_types(self, dfs: pd.DataFrame) -> None:
        """
        Function checks that the target type is acceptable
        """
        if self.type not in ["BINARY", "CAT", "NUM"]:
            raise TypeError(f"{self.type} is not an allowed type for the target")
        if self.type == "CAT":
            assert dfs[self.target].nunique() < self.config["max_target_class"].get(
                int
            ), f"Number of categories {dfs[self.target].nunique()} is too large"

    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
        """
        method for running the main function from TargetAnalyzer
        """
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
        df_clean = df_small[~df_small[self.col].isna()]
        if self.type == "NUM":
            df_clean = df_clean[np.isfinite(df_clean[self.col].values)]  # infinite removed
            if self.config["pct_outliers"].get(float) > 0:
                # remove outliers (only in numeric data)
                df_clean = self.remove_outliers(df_clean)
        self.rate_non_nulls = df_clean.shape[0] / df_small.shape[0]
        return df_clean

    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method removes outliers according to parameter self.pct outliers
        Only applicable for numeric data
        percentile
        """
        pct_outliers = self.config["pct_outliers"].get(float)
        limits = [pct_outliers * 100 / 2, 100 - pct_outliers * 100 / 2]
        pct_val = list(np.percentile(data[self.col], limits))
        data = data[data[self.col] >= pct_val[0]]
        data = data[data[self.col] <= pct_val[1]]
        return data

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
        Do some type changing before starting, object type is not allowed
        """
        # change str to cat
        if data[self.col].dtype.name == "object":
            data[self.col] = data[self.col].astype("category")
        return data

    def run(self, data: pd.DataFrame, result_dict: ResultDict) -> ResultDict:
        """
        Main function for running the column analysis
        """
        if not self.sanity_checks(data):
            return result_dict
        df_small = get_df_small(data, self.col, self.target)
        df_small = self.change_types(df_small)
        df_small = self.get_type(df_small)
        if self.type == "UNIQUE":
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

        fig = plt.figure(figsize=tuple(self.config["figures_size"].get(list)))
        fig.suptitle("Relation between column {} and target".format(self.col), size=16)

        # Scatter for continuous and boxplot for category
        self.plot_var_target_relation(df_clean, cut_series)

        # Relation plot
        self.plot_cut_var_target_relation(df_clean, cut_series)

        rel_fig = plot_360_n0sc0pe()
        result_dict[self.col]["rel_fig"] = rel_fig

        return result_dict

    def plot_cut_var_target_relation(self, df_small: pd.DataFrame, cut_col: pd.Series) -> None:
        """
        Plots of relation between buckets of column and target value
        """
        ax1 = plt.subplot2grid((2, 2), (0, 1))
        # add y label to ax1
        target_type = self.config["target_type"].get(str)
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
                    if not self.config["plot_0_in_binary_target"].get(bool):
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
        target_type: str = self.config["target_type"].get(str)
        cmap: str = self.config["heatmap"]["cmap"].get(str)
        max_scatter_points: int = self.config["max_scatter_points"].get(int)

        if self.type in ["CAT", "BINARY", "DATE"]:
            if target_type == "NUM":
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
                sns.boxplot(
                    x=df_small[self.col],
                    y=df_small[self.target],
                    ax=ax0,
                    orient="h",
                    showfliers=False,
                )
                truncate_labels(ax0, self.config)
        plt.title(" ", fontsize=20)  # necessary so that space is left for main title

    def get_buckets(self, df_small: pd.DataFrame) -> pd.Series:
        """
        Function to cut column into different buckets, for later analysis
        """
        n_breaks = self.config["n_breaks"].get(int)
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
                        df_small[self.col].apply(lambda x: int(x.strftime("%Y%m%d"))),
                        n_breaks,
                        duplicates="drop",
                        precision=self.config["qcut_precision"].get(int),
                    ).cat.remove_unused_categories()

                else:
                    cut_col = pd.qcut(
                        df_small[self.col],
                        n_breaks,
                        duplicates="drop",
                        precision=self.config["qcut_precision"].get(int),
                    ).cat.remove_unused_categories()
        if cut_col.nunique() == 1:
            warning_msg = (
                f"Column {self.col} has only one bucket. This can be caused\n"
                "by a numeric variable being dominated by one number"
            )
            self.log.warning(warning_msg)
        return cut_col

    def calc_explained_variance(self, data: pd.DataFrame, cut_series: pd.Series) -> float:
        """
        Calculate explained variance when binning into n groups.
        It is a proxy of the importance of that variable for a future model.
        The variance is weighted by the rate of non nulls
        """
        assert cut_series.isna().sum() == 0
        assert data.shape[0] == cut_series.shape[0]
        target_type = self.config["target_type"].get(str)
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
        Perform calc_explained_variance for the case of categorical/binary target
        """
        exp_var_list = []
        for target_val in series_target.unique():
            series_target_cat = series_target == target_val
            exp_var_list.append(self.calc_explained_variance_(series_target_cat, cut_series))
        return np.mean(exp_var_list) * self.rate_non_nulls

    @staticmethod
    def calc_explained_variance_(series_target: pd.Series, cut_series: pd.Series) -> float:
        """
        Generic method for calc explained variance
        """
        variance = series_target.var(ddof=0) * series_target.shape[0]

        non_exp_variance = (
            series_target.groupby(cut_series, observed=False).var(ddof=0).values
            * series_target.groupby(cut_series, observed=False).count().values
        ).sum()

        # Handle the case where variance is zero to avoid division by zero
        if variance == 0:
            return 0.0  # If variance is zero, there's no variation to explain
        else:
            return 1 - non_exp_variance / variance


def set_default_params(
    config_: ConfigDict, columns: Optional[List[str]], target: str, data: pd.DataFrame
) -> Tuple[List[str], str]:
    """
    Set default parameters from config_ and set default columns to use
    """
    name_file_out = config_["name_file_out"].get(str)

    if name_file_out == "default":
        timestamp = config_["timestamp"].get(str)
        name_file_out = "targetviz_report_{}.html".format(timestamp)

    # if name does not end in .html append it
    name_file_out = name_file_out if name_file_out.endswith(".html") else name_file_out + ".html"

    if columns is None:
        columns = list(set(data.columns).difference([target]))
    else:
        assert isinstance(columns, Sequence)

    return columns, name_file_out


def base64_image(image: bytes, mime_type: str) -> str:
    """Encode the image for an URL using base64

    Args:
        image: the image
        mime_type: the mime type

    Returns:
        A string starting with "data:{mime_type};base64,"
    """
    base64_data = base64.b64encode(image)
    image_data = quote(base64_data)
    return f"data:{mime_type};base64,{image_data}"


def plot_360_n0sc0pe() -> str:
    """Quickscope the plot to a base64 encoded string.

    Returns:
        A base64 encoded version of the plot in the specified image format.
    """

    image_bytes = BytesIO()
    plt.savefig(
        image_bytes,
        format="png",
    )
    plt.close()
    result_string = base64_image(image_bytes.getvalue(), "image/png")

    return result_string


def targetviz_report(
    data: pd.DataFrame,
    target: str,
    columns: Optional[List[str]] = None,
    output_dir: str = "./",
    **kwargs: Any,
) -> None:
    """
    This function generates a report with plots and statistics showing the relation
    between a target variable and a set of other variables
    :param data: DataFrame where all result_dict is located
    :param target: name of the column to use as target
    :param columns: list of names of the columns to use as dependent variables
    :param output_dir: directory to output result
    :return: No output is returns, instead an html file with all the info and plots is
     generated
    """
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    config.__setitem__("timestamp", timestamp)

    # Add the new default parameter
    config.__setitem__("max_scatter_points", 1000)

    config.set_kwargs(kwargs)
    columns, name_file_out = set_default_params(config, columns, target, data)

    log = create_log(timestamp, output_dir, log_name=name_file_out.replace(".html", ".log"))

    result_dict: ResultDict = {"target": target}

    target_analyzer = TargetAnalyzer(target, config, log)
    data, target_histogram, target_table = target_analyzer.run(data)
    result_dict["target_histogram"] = target_histogram
    result_dict["target_table"] = target_table
    config.__setitem__("target_type", target_analyzer.type)

    total_cols: int = len(columns)
    for i, col in enumerate(columns, start=1):
        log.info(f"({i}/{total_cols}) Analyzing column: {col}")
        col_analyzer = ColumnAnalyzer(col, target, config, log)
        result_dict = col_analyzer.run(data, result_dict)

    render_output(result_dict, columns, output_dir + name_file_out)
