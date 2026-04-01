"""
Utility functions for metrics computation and formatting.

This package contains shared utility functions used across metric computers
and formatters.
"""

from .metric_helpers import (
    validate_model_has_attributes,
    safe_divide,
    clamp_percentage,
)

__all__ = [
    'validate_model_has_attributes',
    'safe_divide',
    'clamp_percentage',
]

