"""Report rendering and output functions for targetviz."""

import os
import zipfile
from collections.abc import Sequence
from typing import List, Optional, Tuple

import pandas as pd
from jinja2 import Template

from targetviz.config import Settings
from targetviz.typedefs import ResultDict
from targetviz.utils import _get_template_path


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


def render_output(result_dict: ResultDict, columns: List[str], name_html: str) -> None:
    """
    Create html file, populate and render file
    """
    # Create html file and render
    base_html = _get_template_path("base.html")

    # fill jinja template with data
    # Use UTF-8 encoding when reading template files
    with open(base_html, "r", encoding="utf-8") as file:
        template = Template(file.read())
    extra_html = _get_template_path("extra_col.html")

    used_cols = sort_cols_by_exp_var(result_dict, columns)
    skipped_variables = result_dict.get("skipped_variables", [])

    # Build TOC data: list of (column_name, explained_var) sorted by importance
    toc_entries = []
    for col in used_cols:
        ev = result_dict[col].get("explained_var", 0)
        toc_entries.append({"name": col, "explained_var": ev})

    html = template.render(
        result_dict=result_dict,
        columns=columns,
        skipped_variables=skipped_variables,
        toc_entries=toc_entries,
    )

    # Use UTF-8 encoding when reading template files
    with open(extra_html, "r", encoding="utf-8") as file:
        extra_template = Template(file.read())

    for col in used_cols:
        html_out = extra_template.render(result_dict=result_dict, column=col)
        html += html_out

    # Close the wrapper div, body and html tags opened in base.html
    html += "\n    </div>\n</body>\n</html>"

    # Check if output should be zipped
    if name_html.endswith(".html.zip"):
        # Create temporary HTML file
        html_filename = name_html[:-4]  # Remove .zip extension
        # Specify UTF-8 encoding when writing the temporary HTML file
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html)

        # Create zip file
        with zipfile.ZipFile(name_html, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(html_filename, os.path.basename(html_filename))

        # Remove temporary HTML file
        os.remove(html_filename)
    else:
        # Specify UTF-8 encoding when writing the final HTML file
        with open(name_html, "w", encoding="utf-8") as f:
            f.write(html)


def set_default_params(
    config_: Settings, columns: Optional[List[str]], target: str, data: pd.DataFrame
) -> Tuple[List[str], str]:
    """
    Set default parameters from config_ and set default columns to use
    """
    name_file_out = config_.name_file_out

    if name_file_out == "default":
        timestamp = config_.timestamp
        name_file_out = "targetviz_report_{}.html".format(timestamp)

    # Check if file should end with html or html.zip
    if not (name_file_out.endswith(".html") or name_file_out.endswith(".html.zip")):
        name_file_out = name_file_out + ".html"

    if columns is None:
        columns = list(set(data.columns).difference([target]))
    else:
        assert isinstance(columns, Sequence)

    return columns, name_file_out
