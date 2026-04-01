"""
Utilization metrics formatter.

TODO: Extract from metric_formatter.py in Phase 2.
"""

from .base_formatter import BaseFormatter


class UtilizationFormatter(BaseFormatter):
    """Formats utilization metrics. TODO: Phase 2"""
    
    def format_utilization_metrics(self, time_step_us: float):
        raise NotImplementedError("Phase 2: Extract from metric_formatter.py")

