"""
Metrics Package

Provides metric computation and formatting for simulation results:
- metric_computer: Computes performance metrics from simulation data
- metric_formatter: Formats metrics for output and visualization

The package is being refactored into modular components:
- computers/: Specialized metric computers
- formatters/: Specialized metric formatters  
- utils/: Shared utilities
- metric_node.py: Core data structure for formatted metrics
"""

# Import from new locations (Phase 1 complete)
from .metric_node import MetricNode

# Import from existing modules (backward compatibility maintained)
from .metric_computer import MetricComputer
from .metric_formatter import MetricFormatter

__all__ = [
    'MetricComputer',
    'MetricFormatter',
    'MetricNode',
]
