"""
Base class for metric formatters.

This module provides the base class that all specialized metric formatters
inherit from, containing common utilities and access to metric computer and global manager.
"""

from typing import Any
from ..metric_node import MetricNode


class BaseFormatter:
    """
    Base class for all metric formatters.
    
    Provides shared access to metric computer and global manager that
    specialized formatters can use.
    
    Attributes:
        mc: The MetricComputer instance
        gm: The GlobalManager instance
    """
    
    def __init__(self, metric_computer: Any, global_manager: Any):
        """
        Initialize the base formatter.
        
        Args:
            metric_computer: MetricComputer instance with computed metrics
            global_manager: GlobalManager instance with simulation state
        """
        self.mc = metric_computer
        self.gm = global_manager
    
    def _create_empty_node(self, section: str, name: str, description: str) -> MetricNode:
        """
        Create an empty MetricNode with a description.
        
        Args:
            section: Section name
            name: Metric name
            description: Description text
        
        Returns:
            MetricNode with description
        """
        return MetricNode(
            section=section,
            name=name,
            description=description
        )
    
    def _create_value_node(self, section: str, name: str, value: Any, unit: str = "") -> MetricNode:
        """
        Create a MetricNode with a single value.
        
        Args:
            section: Section name
            name: Metric name
            value: The metric value
            unit: Unit of measurement
        
        Returns:
            MetricNode with value
        """
        return MetricNode(
            section=section,
            name=name,
            value=value,
            unit=unit
        )
    
    def _create_table_node(self, section: str, name: str, columns: list, rows: list) -> MetricNode:
        """
        Create a MetricNode representing a table.
        
        Args:
            section: Section name
            name: Table name
            columns: List of column names
            rows: List of row dictionaries
        
        Returns:
            MetricNode with table data
        """
        return MetricNode(
            section=section,
            name=name,
            columns=columns,
            rows=rows
        )

