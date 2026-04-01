"""
Compute Simulation Package

Provides compute simulation functionality using multiple backends:
- compute_simulator: Main interface with automatic backend selection and caching
- cimloop_backend: CIMLoop-based simulation for IMC chiplets
- analytical_backend: Analytical model simulation for CMOS chiplets
"""

from .compute_simulator import ComputeSimulator
from .cimloop_backend import CIMLoopBackend
from .analytical_backend import AnalyticalBackend

__all__ = [
    'ComputeSimulator',
    'CIMLoopBackend',
    'AnalyticalBackend',
]
