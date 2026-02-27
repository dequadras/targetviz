"""Main entry point for generating targetviz reports."""

import warnings
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd

from targetviz.analyzers import ColumnAnalyzer, TargetAnalyzer
from targetviz.config import config
from targetviz.report import build_html, set_default_params
from targetviz.report_object import TargetVizReport
from targetviz.typedefs import ResultDict
from targetviz.utils import create_log


def targetviz_report(
    data: pd.DataFrame,
    target: str,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = "./",
    **kwargs: Any,
) -> TargetVizReport:
    """
    Generate a report showing the relation between a target variable and
    other variables.

    :param data: DataFrame containing all data.
    :param target: Name of the column to use as target.
    :param columns: Column names to analyse. Uses all non-target columns
        when *None*.
    :param output_dir: Directory to write the HTML file.  Set to *None*
        to skip file output (useful for notebook-embedded mode).
    :return: A :class:`TargetVizReport` that can be displayed inline in
        a Jupyter notebook or saved later with :meth:`to_file`.
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
    with warnings.catch_warnings():
        # seaborn <0.13 calls groupby without observed=; suppress pandas FutureWarning
        warnings.filterwarnings("ignore", message="observed", category=FutureWarning)
        for i, col in enumerate(columns, start=1):
            log.info(f"({i}/{total_cols}) Analyzing column: {col}")
            col_analyzer = ColumnAnalyzer(col, target, config, log)
            result_dict = col_analyzer.run(data, result_dict)

    # Build the HTML report
    html_content = build_html(result_dict, columns, name_file_out)
    report = TargetVizReport(html_content)

    # Write to file unless output_dir is None (notebook-embedded mode)
    if output_dir is not None:
        report.to_file(output_dir + name_file_out)

    return report
