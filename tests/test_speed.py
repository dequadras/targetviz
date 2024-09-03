"""
Test speed for big datasets
"""

import numpy as np
import pandas as pd
import pytest

from targetviz.main import targetviz_report

SIZE = 1000000


def create_big_dataset():
    """
    Creates a pandas DataFrame with many rows to test speed
    :return: pandas DataFrame
    """
    np.random.seed(42)
    big_df = pd.DataFrame(
        {
            "reg_var": np.random.normal(9, size=SIZE),
            "target": np.random.normal(10, size=SIZE),
        }
    )
    return big_df


def create_big_cat_dataset():
    """
    Creates a pandas DataFrame with many rows to test speed
    :return: pandas DataFrame
    """
    np.random.seed(42)
    n_categories = 9
    big_cat_df = pd.DataFrame(
        {
            "cat_var": np.random.randint(n_categories, size=SIZE),
            "target": np.random.normal(10, size=SIZE),
        }
    )
    big_cat_df["cat_var"] = big_cat_df.cat_var.map({i: f"t{i}" for i in range(n_categories)})
    return big_cat_df


def report_big_df():
    """
    Perform report for big dataframe with continuous variable
    """
    big_df = create_big_dataset()
    targetviz_report(big_df, "target")


def report_big_df_cat():
    """
    Perform report for big dataframe with categorical variable
    """
    big_cat_df = create_big_cat_dataset()
    targetviz_report(big_cat_df, "target")


@pytest.mark.skip(reason="Test is slow and should only be used manually")
def test_speed():
    """
    Main module functionality. Perform report for big dataframes
    """
    report_big_df()
    report_big_df_cat()
