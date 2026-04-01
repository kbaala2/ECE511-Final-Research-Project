"""Simulation summary formatter. TODO: Phase 4"""

from .base_formatter import BaseFormatter

class SummaryFormatter(BaseFormatter):
    def format_simulation_summary(self, main_simulation_duration_seconds: float = 0.0):
        raise NotImplementedError("Phase 4")

