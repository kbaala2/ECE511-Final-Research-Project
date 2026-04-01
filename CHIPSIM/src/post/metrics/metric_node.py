"""
MetricNode data structure for representing metrics in a tree format.

This module provides the core data structure used by the metric formatter
to represent metrics in a hierarchical, structured format.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricNode:
    """
    Represents a metric or group of metrics in a tree structure.
    
    A MetricNode can represent:
    - A simple metric with a single value
    - A structured metric with multiple named values
    - A table with rows and columns
    - A group of child metrics
    
    Attributes:
        section: The section this metric belongs to
        name: The name of the metric
        value: Single value for simple metrics
        values: Dictionary of values for structured metrics
        columns: Column names for table metrics
        rows: List of row dictionaries for table metrics
        description: Description text for empty or explanatory nodes
        children: List of child MetricNode objects
        unit: Unit of measurement
        fmt: Format string for numeric values
        save_children_separately: If True, children are saved as separate files
        children_folder_name: Folder name for separately saved children
    """
    section: str = ""
    name: str = ""
    value: Optional[Any] = None
    values: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Any]]] = None
    description: str = ""
    children: Optional[List["MetricNode"]] = field(default_factory=list)
    unit: str = ""
    fmt: str = ".2f"
    save_children_separately: bool = False
    children_folder_name: Optional[str] = None

