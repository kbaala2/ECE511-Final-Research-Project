"""
Core Package

Provides core simulation components for the chiplet simulator:
- chiplet: Individual chiplet hardware representation
- comm_types: Communication and phase type definitions
- mapped_model: Model mapping and phase management
- model_processor: Model definition processing and metric calculation
- system: System-level chiplet cluster management
- traffic_calculator: Traffic calculation for communication phases
"""

from .chiplet import Chiplet
from .comm_types import (
    Phase, PhaseInstance, PhaseState, TrafficMatrixDict,
    ComputePhase, ActivationCommPhase, WeightLoadingPhase,
    CommunicationPhase,
    create_compute_phase, create_activation_comm_phase, create_weight_loading_phase
)
from .mapped_model import MappedModel
from .model_processor import ModelProcessor
from .system import System
from .traffic_calculator import TrafficCalculator

__all__ = [
    'Chiplet',
    'Phase',
    'PhaseInstance',
    'PhaseState',
    'TrafficMatrixDict',
    'ComputePhase',
    'ActivationCommPhase',
    'WeightLoadingPhase',
    'CommunicationPhase',
    'create_compute_phase',
    'create_activation_comm_phase',
    'create_weight_loading_phase',
    'MappedModel',
    'ModelProcessor',
    'System',
    'TrafficCalculator',
]
