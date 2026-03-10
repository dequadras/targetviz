"""Type definitions for targetviz."""

from datetime import datetime
from typing import Any, Callable, Dict, TypedDict, Union

ConfigDict = Any  # Kept for backward compatibility; prefer using Settings directly
ResultDict = Dict[str, Dict[str, Union[str, float, Dict[str, Any]]]]


class DescParams(TypedDict):
    full_samp: bool
    is_cat: bool
    is_date: bool
    formatter: Callable[[Union[float, datetime]], str]
