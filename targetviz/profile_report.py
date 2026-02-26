"""Main entry point for generating targetviz reports."""

from datetime import datetime
from typing import Any, List, Optional

import pandas as pd

from targetviz.analyzers import ColumnAnalyzer, TargetAnalyzer
from targetviz.config import config
from targetviz.report import render_output, set_default_params
from targetviz.typedefs import ResultDict
from targetviz.utils import create_log


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
    config.timestamp = timestamp

    # Apply kwargs which will override any defaults
    config.set_kwargs(kwargs)

    columns, name_file_out = set_default_params(config, columns, target, data)

    log = create_log()

    result_dict: ResultDict = {"target": target, "skipped_variables": []}

    target_analyzer = TargetAnalyzer(target, config, log)
    data, target_histogram, target_table = target_analyzer.run(data)
    result_dict["target_histogram"] = target_histogram
    result_dict["target_table"] = target_table
    config.target_type = target_analyzer.type

    total_cols: int = len(columns)
    for i, col in enumerate(columns, start=1):
        log.info(f"({i}/{total_cols}) Analyzing column: {col}")
        col_analyzer = ColumnAnalyzer(col, target, config, log)
        result_dict = col_analyzer.run(data, result_dict)

    render_output(result_dict, columns, output_dir + name_file_out)
