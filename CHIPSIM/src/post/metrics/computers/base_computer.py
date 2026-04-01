"""
Base class for metric computers with shared utilities and state.

This module provides the base class that all specialized metric computers
inherit from, containing common state and utility methods.
"""

from typing import Any, Dict, Set


class BaseMetricComputer:
    """
    Base class for all metric computers.
    
    Provides shared state and utilities that specialized computers can use.
    All metric computers should inherit from this class.
    
    Attributes:
        retired_mapped_models: Dictionary of completed mapped models
        global_time_us: Final global time of the simulation in microseconds
        num_chiplets: Total number of chiplets in the system
        all_chiplets: Set of all chiplet IDs used across all models
    """
    
    def __init__(self, retired_mapped_models: Dict[int, Any], global_time_us: float, 
                 num_chiplets: int, all_chiplets: Set[int] = None):
        """
        Initialize the base computer with shared state.
        
        Args:
            retired_mapped_models: Dictionary of completed mapped models
            global_time_us: Final global time of the simulation in microseconds
            num_chiplets: Total number of chiplets in the system
            all_chiplets: Pre-computed set of all chiplets (optional, will compute if not provided)
        """
        self.retired_mapped_models = retired_mapped_models
        self.global_time_us = global_time_us
        self.total_simulation_time_us = global_time_us  # Alias for compatibility
        self.num_chiplets = num_chiplets
        
        # Use provided chiplets or extract from models
        if all_chiplets is not None:
            self.all_chiplets = all_chiplets
        else:
            self.all_chiplets: Set[int] = set()
            for model in retired_mapped_models.values():
                if hasattr(model, 'get_chiplets_used'):
                    self.all_chiplets.update(model.get_chiplets_used())
    
    def _validate_model_attributes(self, model: Any, required_attrs: list, model_id: str = "Model") -> None:
        """
        Validate that a model has all required attributes.
        
        Args:
            model: The model to validate
            required_attrs: List of required attribute names
            model_id: Identifier for error messages
        
        Raises:
            AssertionError: If any required attribute is missing
        """
        model_name = getattr(model, 'model_name', model_id)
        for attr in required_attrs:
            assert hasattr(model, attr), \
                   f"{model_id} ({model_name}) is missing '{attr}' attribute. " \
                   f"This is required for metric computation."
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero.
        
        Args:
            numerator: The numerator
            denominator: The denominator  
            default: Value to return if denominator is zero
        
        Returns:
            Result of division or default value
        """
        if denominator == 0 or denominator is None:
            return default
        try:
            return numerator / denominator
        except (ZeroDivisionError, TypeError):
            return default
    
    def _clamp_percentage(self, value: float) -> float:
        """
        Clamp a value to [0, 100] percentage range.
        
        Args:
            value: The value to clamp
        
        Returns:
            Clamped value between 0.0 and 100.0
        """
        if value is None:
            return 0.0
        return max(0.0, min(100.0, value))
    
    def _validate_latency(self, latency: Any) -> float:
        """
        Validate and sanitize a latency value.
        
        Args:
            latency: The latency value to validate
        
        Returns:
            Valid latency (>= 0.0)
        """
        if not isinstance(latency, (int, float)):
            return 0.0
        return max(0.0, float(latency))
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get computed results from this computer.
        
        Should be overridden by subclasses to return their specific metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        return {}

