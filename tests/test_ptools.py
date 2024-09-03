"""
Implement test for targetviz report
"""

from time import sleep
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
from sklearn import datasets

from targetviz.config import config
from targetviz.main import BaseAnalyzer, ColumnAnalyzer, targetviz_report


def sklearn_to_df(sklearn_dataset):
    """
    Create pandas DataFrame from sklearn dataset
    :param sklearn_dataset: result of loading datasets with sklearn, for
        example: sklearn.datasets.fetch_california_housing()
    :return: pandas DataFrame with variables and target
    """
    data = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    data["target"] = pd.Series(sklearn_dataset.target)
    return data


def create_cat_dataset():
    """
    Creates a pandas DataFrame with categorical variables to use in some test
    :return: pandas DataFrame
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "cat_var": np.random.randint(9, size=100),
            "target": np.random.normal(10, size=100),
        }
    )
    data["cat_var"] = data.cat_var.map(
        {
            0: "t0",
            1: "t1",
            2: "t2",
            3: "t3",
            4: "t4",
            5: "t5",
            6: "t6",
            7: "t7",
            8: "t8",
        }
    )

    return data


def create_dt_dataset():
    """
    Create pandas DataFrame with some date values, and both numeric and categorical
        variables
    :return: pandas DataFrame
    """
    np.random.seed(42)
    starting_date = pd.Timestamp(2019, 7, 1)
    data = pd.DataFrame(
        {
            "date": (
                starting_date + pd.to_timedelta(np.random.randint(1, 100, size=100), unit="d")
            ),
            "target": np.random.normal(10, size=100),
        }
    )

    return data


def create_nan_dataset():
    """
    Create pandas DataFrame with some nan values, and both numeric and categorical
        variables
    :return: pandas DataFrame
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "cat_var": np.random.randint(9, size=100),
            "reg_var": np.random.normal(5, size=100),
            "target": np.random.normal(10, size=100),
        }
    )
    data["cat_var"] = data.cat_var.map(
        {
            0: "t0",
            1: "t1",
            2: "t2",
            3: "t3",
            4: "t4",
            5: "t5",
            6: "t6",
            7: "t7",
            8: "t8",
        }
    )
    data.loc[list(np.random.randint(100, size=10)), "cat_var"] = np.nan
    data.loc[list(np.random.randint(100, size=10)), "reg_var"] = np.nan

    return data


def create_df_unique():
    """
    Function gets dataset with only unique values, to test that pd.qcut is not a problem

    :return:
        (pd.DataFrame)
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "target": (np.random.uniform(size=10000) > 0.1) * 1,
            "pred": np.random.choice(
                np.array([-999, 0, 1.1, 3.2]),
                replace=True,
                p=np.array([0.95, 0.02, 0.02, 0.01]),
            ),
        }
    )
    return data


def test_targetviz():
    """
    Test cases for targetviz, asserting that no error arises during runtime
    """
    df_cal = sklearn_to_df(datasets.fetch_california_housing())
    target_col = "target"
    assert targetviz_report(df_cal, target_col) is None

    data = create_cat_dataset()
    assert targetviz_report(data, "target") is None

    df_nan = create_nan_dataset()
    assert targetviz_report(df_nan, "target", pct_outliers=0.05) is None

    df_dt = create_dt_dataset()
    assert targetviz_report(df_dt, "target", pct_outliers=0.05) is None

    df_unique = create_df_unique()
    assert targetviz_report(df_unique, "target", output_dir="./") is None


def test_calc_explained_var():
    """
    Test some cases for explained variance
    """
    df_cal = sklearn_to_df(datasets.fetch_california_housing())
    target_col = "target"
    n_breaks = 5

    # Test 1
    col = "AveRooms"
    new_col = pd.qcut(df_cal[col], n_breaks, duplicates="drop").cat.remove_unused_categories()

    config_ = config
    config_.__setitem__("target_type", "NUM")
    col_analysis = ColumnAnalyzer(Mock, target_col, config_, Mock)
    col_analysis.rate_non_nulls = 1
    np.testing.assert_almost_equal(
        col_analysis.calc_explained_variance(df_cal, new_col), 0.1240, decimal=3
    )

    # Test 2
    col_analysis = ColumnAnalyzer(Mock, target_col, config_, Mock)
    col_analysis.rate_non_nulls = 1
    col = "HouseAge"
    new_col = pd.qcut(df_cal[col], n_breaks, duplicates="drop").cat.remove_unused_categories()
    np.testing.assert_almost_equal(
        col_analysis.calc_explained_variance(df_cal, new_col), 0.0103, decimal=3
    )

    # Test 3
    col_analysis.rate_non_nulls = 0.5
    np.testing.assert_almost_equal(
        col_analysis.calc_explained_variance(df_cal, new_col), 0.00519, decimal=3
    )


def test_get_desc():
    """
    Test get_desc function, that is in charge of returning statistic description of
        variables
    """
    np.random.seed(42)
    quantiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]

    series = pd.Series(np.random.uniform(0, 1, size=20))
    expected_result = [
        20,
        20,
        0,
        "0.00%",
        "0.97",
        "0.02",
        "0.46",
        "0.40",
        "0.02",
        "0.31",
        "0.03",
        "0.06",
        "0.18",
        "0.71",
        "0.95",
        "0.97",
    ]

    base = BaseAnalyzer(Mock, Mock, MagicMock(), Mock)
    result = base.get_desc(series, full_samp=True, quantiles=quantiles)
    assert expected_result == result

    series[list(np.random.randint(20, size=3))] = np.nan
    expected_result2 = [
        20,
        17,
        3,
        "0.15%",
        "0.97",
        "0.02",
        "0.48",
        "0.43",
        "0.02",
        "0.31",
        "0.03",
        "0.05",
        "0.21",
        "0.73",
        "0.95",
        "0.97",
    ]
    assert expected_result2 == base.get_desc(series, full_samp=True, quantiles=quantiles)
    expected_result3 = [
        20,
        17,
        "-",
        "-",
        "0.97",
        "0.02",
        "0.48",
        "0.43",
        "0.02",
        "0.31",
        "0.03",
        "0.05",
        "0.21",
        "0.73",
        "0.95",
        "0.97",
    ]
    assert expected_result3 == base.get_desc(series, full_samp=False, quantiles=quantiles)


def test_numeric_name():
    sleep(1)  # make sure folder names change since last test
    df_names_num = pd.DataFrame({"pred": [0, 1, 2, 3], "0": [2, 3, 4, 5], 0: [7, 8, 9, 0]})
    targetviz_report(df_names_num, "pred", n_breaks=2)


def test_nullable_types():
    # Test with Int (nullable int)
    df_int = pd.DataFrame(
        {"target": [1, 2, 3, 4, pd.NA], "int_col": pd.array([1, 2, 3, pd.NA, 5], dtype="Int64")}
    )
    assert targetviz_report(df_int, "target") is None

    # Test with Float (nullable float)
    df_float = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0, pd.NA],
            "float_col": pd.array([1.1, 2.2, 3.3, pd.NA, 5.5], dtype="Float64"),
        }
    )
    assert targetviz_report(df_float, "target") is None

    # Test with Bool (nullable boolean)
    df_bool = pd.DataFrame(
        {
            "target": [True, False, True, False, pd.NA],
            "bool_col": pd.array([True, False, True, pd.NA, False], dtype="boolean"),
        }
    )
    assert targetviz_report(df_bool, "target") is None

    # Test with mixed types
    df_mixed = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "int_col": pd.array([1, 2, 3, pd.NA, 5], dtype="Int64"),
            "float_col": pd.array([1.1, 2.2, 3.3, pd.NA, 5.5], dtype="Float64"),
            "bool_col": pd.array([True, False, True, pd.NA, False], dtype="boolean"),
        }
    )
    assert targetviz_report(df_mixed, "target") is None
