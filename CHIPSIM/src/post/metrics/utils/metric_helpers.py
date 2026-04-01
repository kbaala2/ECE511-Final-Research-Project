"""
Utility functions shared across metric computers and formatters.

This module provides common helper functions for validation, safe arithmetic,
and other operations used throughout the metrics module.
"""

from typing import Any, List, Optional


def validate_model_has_attributes(model: Any, required_attrs: List[str], model_identifier: str = "Model") -> None:
    """
    Validate that a model object has all required attributes.
    
    Args:
        model: The model object to validate
        required_attrs: List of required attribute names
        model_identifier: String identifier for the model (for error messages)
    
    Raises:
        AssertionError: If any required attribute is missing
    """
    model_name = getattr(model, 'model_name', model_identifier)
    model_idx = getattr(model, 'model_idx', 'Unknown')
    
    for attr in required_attrs:
        assert hasattr(model, attr), \
               f"{model_identifier} {model_idx} ({model_name}) is missing '{attr}' attribute. " \
               f"This is required for metric computation."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: The value to return if denominator is zero or invalid
    
    Returns:
        numerator / denominator if denominator != 0, else default
    """
    if denominator is None or denominator == 0:
        return default
    try:
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def clamp_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Clamp a percentage value to a valid range.
    
    Args:
        value: The value to clamp
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 100.0)
    
    Returns:
        Clamped value between min_val and max_val
    """
    if value is None:
        return 0.0
    return max(min_val, min(max_val, value))


def get_chiplets_from_models(retired_models: dict) -> set:
    """
    Extract all unique chiplet IDs from a collection of retired models.
    
    Args:
        retired_models: Dictionary of model_idx -> model mappings
    
    Returns:
        Set of all chiplet IDs used across all models
    """
    all_chiplets = set()
    for model in retired_models.values():
        if hasattr(model, 'get_chiplets_used'):
            all_chiplets.update(model.get_chiplets_used())
    return all_chiplets


def validate_latency_value(latency: Any) -> float:
    """
    Validate and sanitize a latency value.
    
    Converts invalid or negative latencies to 0.0.
    
    Args:
        latency: The latency value to validate
    
    Returns:
        Valid latency value (>= 0.0)
    """
    if not isinstance(latency, (int, float)):
        return 0.0
    return max(0.0, float(latency))


def sum_dict_values(data_dict: Optional[dict], key_filter: Optional[set] = None) -> float:
    """
    Sum values from a dictionary, optionally filtering by keys.
    
    Args:
        data_dict: Dictionary to sum values from
        key_filter: Optional set of keys to include (None = all keys)
    
    Returns:
        Sum of values (0.0 if dict is None or empty)
    """
    if not data_dict:
        return 0.0
    
    total = 0.0
    for key, value in data_dict.items():
        if key_filter is None or key in key_filter:
            if isinstance(value, (int, float)) and value > 0:
                total += value
    
    return total


def format_time_us(time_us: float, precision: int = 2) -> str:
    """
    Format a time value in microseconds with appropriate unit.
    
    Args:
        time_us: Time in microseconds
        precision: Number of decimal places
    
    Returns:
        Formatted string with appropriate unit (μs, ms, s)
    """
    if time_us < 0:
        return "N/A"
    elif time_us < 1000:
        return f"{time_us:.{precision}f} μs"
    elif time_us < 1_000_000:
        return f"{time_us / 1000:.{precision}f} ms"
    else:
        return f"{time_us / 1_000_000:.{precision}f} s"


def format_energy_uj(energy_uj: float, precision: int = 2) -> str:
    """
    Format an energy value in microjoules with appropriate unit.
    
    Args:
        energy_uj: Energy in microjoules
        precision: Number of decimal places
    
    Returns:
        Formatted string with appropriate unit (μJ, mJ, J)
    """
    if energy_uj < 0:
        return "N/A"
    elif energy_uj < 1000:
        return f"{energy_uj:.{precision}f} μJ"
    elif energy_uj < 1_000_000:
        return f"{energy_uj / 1000:.{precision}f} mJ"
    else:
        return f"{energy_uj / 1_000_000:.{precision}f} J"

