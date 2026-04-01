"""
Post-Processing Package

Provides post-processing and visualization utilities for simulation results:
- cross_sim_processor: Cross-simulation analysis and comparison
- metrics: Metric computation and formatting
- output_manager: Manages simulation output formatting and file operations
- simulation_plotter: Creates plots for simulation metrics
- temporal_filter: Filters simulation data by time periods
- visualize_mapping: Visualizes chiplet system mappings
"""

from .cross_sim_processor import CrossSimProcessor
from .metrics import MetricComputer, MetricFormatter, MetricNode
from .output_manager import OutputManager
from .simulation_plotter import SimulationPlotter
from .temporal_filter import TemporalFilter
from .visualize_mapping import ChipletVisualizer

__all__ = [
    'CrossSimProcessor',
    'MetricComputer',
    'MetricFormatter',
    'MetricNode',
    'OutputManager',
    'SimulationPlotter',
    'TemporalFilter',
    'ChipletVisualizer',
]
