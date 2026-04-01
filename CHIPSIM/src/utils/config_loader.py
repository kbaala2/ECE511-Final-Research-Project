#!/usr/bin/env python3

import os
import yaml
from typing import Dict, Any


def validate_config_structure(config: Dict[str, Any], config_file: str) -> None:
    """
    Validate that the configuration has the expected nested structure.
    
    Args:
        config: Configuration dictionary to validate
        config_file: Path to config file (for error messages)
        
    Raises:
        ValueError: If the configuration structure is invalid
    """
    # Check for required top-level keys
    if 'simulation' not in config:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required 'simulation' key.\n"
            f"Expected structure:\n"
            f"  simulation:\n"
            f"    input_files: ...\n"
            f"    core_settings: ...\n"
            f"    ...\n"
            f"  post_processing: ..."
        )
    
    if 'post_processing' not in config:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required 'post_processing' key.\n"
            f"Expected structure:\n"
            f"  simulation: ...\n"
            f"  post_processing:\n"
            f"    warmup_period_us: ...\n"
            f"    generate_plots: ..."
        )
    
    # Check for required simulation subsections
    required_subsections = ['input_files', 'core_settings', 'hardware_parameters']
    missing_subsections = [s for s in required_subsections if s not in config['simulation']]
    
    if missing_subsections:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required simulation subsection(s): {', '.join(missing_subsections)}\n"
            f"Expected subsections under 'simulation':\n"
            f"  - input_files\n"
            f"  - core_settings\n"
            f"  - hardware_parameters\n"
            f"  - gem5_parameters (optional)\n"
            f"  - dsent_parameters (optional)"
        )


def load_config(config_file: str = "configs/experiments/config_1.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_file: Path to config file, defaults to "configs/experiments/config_1.yaml" 
                    Can be:
                    - Relative path from project root (e.g., "configs/experiments/config_2.yaml")
                    - Just experiment name (e.g., "config_2" - will look in configs/experiments/)
                    - Absolute path
        
    Returns:
        Dictionary containing configuration values with nested structure
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
    """
    # Handle experiment name shortcut (e.g., "config_2" -> "configs/experiments/config_2.yaml")
    if not config_file.endswith('.yaml') and not os.path.isabs(config_file):
        config_file = f"configs/experiments/{config_file}.yaml"
    
    # If relative path, make it relative to the project root
    if not os.path.isabs(config_file):
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(script_dir, config_file)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate the configuration structure
    validate_config_structure(config, config_file)
    
    return config
