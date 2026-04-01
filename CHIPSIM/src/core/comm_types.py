"""
Communication and Phase Type Definitions

This module defines the phase class hierarchy for the chiplet simulator,
including abstract base classes and concrete implementations for different
phase types (compute, activation communication, weight loading, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

# Type alias for traffic matrix representation
TrafficMatrixDict = Dict[int, Dict[int, int]]

# Phase state enumeration
class PhaseState(IntEnum):
    NOT_STARTED = 0
    RUNNING = 1
    COMPLETE = 2


# ============================================================================
# Abstract Base Classes
# ============================================================================

@dataclass
class Phase(ABC):
    """Abstract base class for all phase types in the simulation"""
    phase_id: int  # Unique identifier for this phase
    model_idx: int  # Index of the model this phase belongs to
    dependencies: List[int] = field(default_factory=list)  # Phase IDs that must complete first
    
    @abstractmethod
    def get_phase_type_name(self) -> str:
        """Return a string identifier for this phase type"""
        pass
    
    @abstractmethod
    def can_generate_traffic(self) -> bool:
        """Check if this phase type generates network traffic"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.phase_id}, model={self.model_idx})"


# ============================================================================
# Compute Phase
# ============================================================================

@dataclass
class ComputePhase(Phase):
    """Phase representing computation on chiplets"""
    layer_idx: int = -1  # Index of the layer being computed (default -1 for compatibility)
    chiplet_assignments: Optional[Dict[int, float]] = field(default=None)  # {chiplet_id: percentage}
    
    def get_phase_type_name(self) -> str:
        return "COMPUTE"
    
    def can_generate_traffic(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return f"ComputePhase(id={self.phase_id}, layer={self.layer_idx})"


# ============================================================================
# Communication Phase Base Class
# ============================================================================

@dataclass
class CommunicationPhase(Phase):
    """Abstract base class for all communication phases"""
    traffic: Optional[TrafficMatrixDict] = field(default=None)
    traffic_source: str = field(default="compute")  # "compute" or "io"
    
    def can_generate_traffic(self) -> bool:
        return True
    
    @abstractmethod
    def get_layer_indices(self) -> List[int]:
        """Get the layer indices involved in this communication"""
        pass


# ============================================================================
# Concrete Communication Phase Types
# ============================================================================

@dataclass
class ActivationCommPhase(CommunicationPhase):
    """Phase for activation communication between layers"""
    layer_idx: int = -1  # Source layer index (default -1 for compatibility)
    
    def get_phase_type_name(self) -> str:
        return "ACTIVATION_COMM"
    
    def get_layer_indices(self) -> List[int]:
        return [self.layer_idx]
    
    def __repr__(self) -> str:
        return f"ActivationCommPhase(id={self.phase_id}, layer={self.layer_idx})"


@dataclass
class WeightLoadingPhase(CommunicationPhase):
    """Phase for weight loading from I/O chiplets"""
    layers_to_load: List[int] = field(default_factory=list)  # Layer indices to load weights for
    
    def __post_init__(self):
        """Ensure traffic source is always 'io' for weight loading"""
        self.traffic_source = "io"
    
    def get_phase_type_name(self) -> str:
        return "WEIGHT_LOADING_COMM"
    
    def get_layer_indices(self) -> List[int]:
        return self.layers_to_load
    
    def is_combined(self) -> bool:
        """Check if this phase loads weights for multiple layers"""
        return len(self.layers_to_load) > 1
    
    def __repr__(self) -> str:
        if self.is_combined():
            return f"WeightLoadingPhase(id={self.phase_id}, layers={self.layers_to_load})"
        else:
            return f"WeightLoadingPhase(id={self.phase_id}, layer={self.layers_to_load[0] if self.layers_to_load else 'none'})"


# ============================================================================
# Runtime Phase Instance
# ============================================================================

@dataclass
class PhaseInstance:
    """Runtime instance of a phase for a specific input"""
    phase_id: int
    input_idx: int
    state: PhaseState
    start_time_us: float = -1.0
    completion_time_us: float = -1.0
    latency_us: float = 0.0
    energy_fj: float = 0.0
    
    # For communication phases - dynamic tracking
    scaled_traffic: Optional[TrafficMatrixDict] = field(default=None)
    percent_complete: float = 0.0
    latency_history: List[tuple] = field(default_factory=list)  # [(time, latency)]
    is_active: bool = False
    
    def __repr__(self) -> str:
        return f"PhaseInstance(phase={self.phase_id}, input={self.input_idx}, state={self.state.name})"


# ============================================================================
# Factory Functions
# ============================================================================

def create_compute_phase(phase_id: int, model_idx: int, layer_idx: int,
                        dependencies: Optional[List[int]] = None,
                        chiplet_assignments: Optional[Dict[int, float]] = None) -> ComputePhase:
    """Factory function to create a compute phase"""
    return ComputePhase(
        phase_id=phase_id,
        model_idx=model_idx,
        dependencies=dependencies or [],
        layer_idx=layer_idx,
        chiplet_assignments=chiplet_assignments
    )


def create_activation_comm_phase(phase_id: int, model_idx: int, layer_idx: int,
                                 dependencies: Optional[List[int]] = None) -> ActivationCommPhase:
    """Factory function to create an activation communication phase"""
    return ActivationCommPhase(
        phase_id=phase_id,
        model_idx=model_idx,
        dependencies=dependencies or [],
        layer_idx=layer_idx,
        traffic_source="compute"
    )


def create_weight_loading_phase(phase_id: int, model_idx: int, layers_to_load: List[int],
                                dependencies: Optional[List[int]] = None) -> WeightLoadingPhase:
    """Factory function to create a weight loading phase"""
    return WeightLoadingPhase(
        phase_id=phase_id,
        model_idx=model_idx,
        dependencies=dependencies or [],
        layers_to_load=layers_to_load
    )