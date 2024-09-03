"""
Tests corresponding to target variables that are multiclass
"""

import numpy as np
import pandas as pd
from sklearn import datasets

from targetviz.main import targetviz_report


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


def test_iris():
    """
    Test report for iris dataset, the target is multiclass
    """
    iris = datasets.load_iris()
    iris_df = sklearn_to_df(iris)
    iris_df["target"] = iris.target_names[iris_df.target]

    targetviz_report(iris_df, "target")


def test_cancer():
    """
    Test report for breast cancer dataset, the target is binary
    """
    breast_cancer = datasets.load_breast_cancer()
    breast_cancer_df = sklearn_to_df(breast_cancer)
    breast_cancer_df["target"] = breast_cancer.target_names[breast_cancer_df.target]

    targetviz_report(breast_cancer_df, "target", columns=["worst area"])


def test_heatmap():
    df = pd.DataFrame(
        {
            "target": np.random.choice(["tar1", "tar2", "tar3"], size=100),
            "cat_var": np.random.choice(["var1", "var2", "var3"], size=100),
        }
    )
    targetviz_report(df, "target")
