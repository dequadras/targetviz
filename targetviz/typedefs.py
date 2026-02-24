"""Type definitions for targetviz."""

from datetime import datetime
from typing import Any, Callable, Dict, List, TypedDict, Union

ConfigDict = Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]
ResultDict = Dict[str, Dict[str, Union[str, float, Dict[str, Any]]]]


class DescParams(TypedDict):
    full_samp: bool
    is_cat: bool
    is_date: bool
    formatter: Callable[[Union[float, datetime]], str]
