"""
Compute Simulator Module

Main interface for compute simulation across different chiplet types.
Automatically selects the appropriate backend (CIMLoop for IMC, Analytical for CMOS)
based on chiplet type and handles caching of simulation results.
"""

import time
from src.managers.cache_manager import CacheManager
from .cimloop_backend import CIMLoopBackend
from .analytical_backend import AnalyticalBackend

DEBUG = False


class ComputeSimulator:
    """
    Handles compute simulation using multiple backends with caching.
    
    Automatically selects the appropriate simulation backend based on chiplet type:
    - IMC chiplets use CIMLoopBackend (YAML-based simulation)
    - CMOS chiplets use AnalyticalBackend (MAC-based analytical model)
    """
    
    def __init__(self, 
                 cache_file="compute_cache.pkl", 
                 clear_cache=False):
        """
        Initialize the compute simulator with caching options.
        
        Args:
            cache_file (str): Path to the cache file
            clear_cache (bool): Whether to clear the existing cache
        """
        # Initialize the cache manager
        self.cache_manager = CacheManager(cache_file, clear_cache)
        
        # Initialize backends
        self.cimloop_backend = CIMLoopBackend()
        self.analytical_backend = AnalyticalBackend()
    
    def _normalize_yaml_config(self, yaml_config):
        """
        Normalize the YAML configuration for consistent cache keys.
        
        Args:
            yaml_config (str): The original YAML configuration
            
        Returns:
            str: Normalized YAML configuration
        """
        # Split the YAML into lines
        lines = yaml_config.split('\n')
        
        # Filter out or normalize lines that contain chiplet-specific information
        normalized_lines = []
        for line in lines:
            # Skip lines with notes that contain chiplet-specific information
            if "notes:" in line and "Partitioned for Chiplet" in line:
                normalized_lines.append("  notes: NORMALIZED")
            else:
                normalized_lines.append(line)
        
        # Rejoin the lines
        return '\n'.join(normalized_lines)
    
    def _compute_cache_key_for_analytical(self, partitioned_layer, chiplet_type, batch_size):
        """
        Compute cache key for analytical model simulations.
        
        Args:
            partitioned_layer (dict): Partitioned layer with total_macs
            chiplet_type (str): Type of chiplet
            batch_size (int): Batch size
            
        Returns:
            str: Cache key
        """
        total_macs = partitioned_layer.get('total_macs', 0)
        description = partitioned_layer.get('description', 'unknown')
        percentage = partitioned_layer.get('percentage', 100)
        
        # Include critical chiplet params to avoid collisions across CMOS configs
        macs_per_second = partitioned_layer.get('macs_per_second')
        energy_per_mac = partitioned_layer.get('energy_per_mac')
        # Fall back to provided chiplet type defaults if not in layer
        # These will be embedded by caller if desired; otherwise, leave None
        key_components = [
            f"analytical",
            f"chiplet_type={chiplet_type}",
            f"macs={total_macs:.0f}",
            f"desc={description}",
            f"pct={percentage:.2f}",
            f"batch={batch_size}",
            f"mps={macs_per_second}",
            f"e_per_mac={energy_per_mac}"
        ]
        return "|".join(key_components)

    def simulate_compute(self, 
                        model_name, 
                        layer_idx, 
                        chiplet_id, 
                        partitioned_layer, 
                        num_layers, 
                        model_idx,
                        chiplet_type,
                        compute_type,
                        chiplet_params,
                        batch_size=1):
        """
        Simulate the compute of a single model chunk.
        Automatically selects backend based on compute_type.
        Uses caching to avoid redundant simulations.
        
        Args:
            model_name (str): Name of the model (e.g., 'ResNet18')
            layer_idx (int): Index of the layer in the model
            chiplet_id (int): ID of the chiplet to simulate on
            partitioned_layer (dict): Partitioned layer definition
            num_layers (int): Total number of layers in the model
            model_idx (int): Index of the model in the workload
            chiplet_type (str): Type of the chiplet (e.g., 'Standard', 'CMOS_Compute')
            compute_type (str): Compute type ('IMC', 'CMOS', or 'IO')
            chiplet_params (dict): Chiplet parameters from CHIPLET_TYPES
            batch_size (int): Batch size for simulation
            
        Returns:
            dict: Simulation results containing latency_us, energy_fj, cycles
            
        Raises:
            ValueError: If compute_type is invalid or mismatched with chiplet capabilities
        """
        # Track the time for the simulation
        sim_start_time = time.time()
        
        # Validate compute type
        if compute_type not in ['IMC', 'CMOS', 'IO']:
            raise ValueError(f"Invalid compute_type '{compute_type}' for chiplet {chiplet_id}. Must be 'IMC', 'CMOS', or 'IO'")
        
        # IO chiplets should not perform compute
        if compute_type == 'IO':
            raise ValueError(f"Cannot simulate compute on I/O chiplet {chiplet_id}")
        
        # Select backend based on compute type
        if compute_type == 'IMC':
            # Use CIMLoop backend for IMC chiplets
            result = self._simulate_with_cimloop(
                model_name=model_name,
                layer_idx=layer_idx,
                chiplet_id=chiplet_id,
                partitioned_layer=partitioned_layer,
                num_layers=num_layers,
                chiplet_type=chiplet_type,
                batch_size=batch_size
            )
        elif compute_type == 'CMOS':
            # Use Analytical backend for CMOS chiplets
            result = self._simulate_with_analytical(
                partitioned_layer=partitioned_layer,
                chiplet_id=chiplet_id,
                chiplet_type=chiplet_type,
                chiplet_params=chiplet_params,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unsupported compute_type '{compute_type}' for chiplet {chiplet_id}")
        
        # Calculate simulation duration
        sim_duration = time.time() - sim_start_time
        print(f"⏱️  Compute simulation completed in {sim_duration:.3f} seconds")
        
        return result
    
    def _simulate_with_cimloop(self, model_name, layer_idx, chiplet_id, partitioned_layer, 
                               num_layers, chiplet_type, batch_size):
        """
        Simulate using CIMLoop backend (for IMC chiplets).
        
        Args:
            model_name (str): Name of the model
            layer_idx (int): Layer index
            chiplet_id (int): Chiplet ID
            partitioned_layer (dict): Partitioned layer definition
            num_layers (int): Total number of layers
            chiplet_type (str): Chiplet type name
            batch_size (int): Batch size
            
        Returns:
            dict: Simulation results
        """
        # Generate YAML config
        yaml_config = self.cimloop_backend.generate_yaml_config(
            model_name=model_name,
            layer_idx=layer_idx,
            chiplet_id=chiplet_id,
            partitioned_layer=partitioned_layer,
            num_layers=num_layers
        )
        
        if not yaml_config:
            raise RuntimeError(f"Failed to generate YAML config for layer {layer_idx}, chiplet {chiplet_id}")
        
        # Check cache
        normalized_yaml_config = self._normalize_yaml_config(yaml_config)
        cache_key = self.cache_manager.compute_cache_key(
            yaml_config=normalized_yaml_config, 
            chiplet_type=chiplet_type, 
            batch_size=batch_size
        )
        
        # If cached, return the cached result
        if self.cache_manager.has_result(cache_key):
            print(f"🔄 Using cached CIMLoop result for chiplet {chiplet_id}")
            return self.cache_manager.get_result(cache_key)
        
        # Run simulation
        result = self.cimloop_backend.simulate(
            yaml_config=yaml_config,
            chiplet_id=chiplet_id,
            chiplet_type=chiplet_type,
            batch_size=batch_size
        )
        
        # Cache the result
        self.cache_manager.store_result(cache_key, result)
        
        return result
    
    def _simulate_with_analytical(self, partitioned_layer, chiplet_id, chiplet_type, 
                                  chiplet_params, batch_size):
        """
        Simulate using Analytical backend (for CMOS chiplets).
        
        Args:
            partitioned_layer (dict): Partitioned layer with total_macs
            chiplet_id (int): Chiplet ID
            chiplet_type (str): Chiplet type name
            chiplet_params (dict): Chiplet parameters
            batch_size (int): Batch size
            
        Returns:
            dict: Simulation results
        """
        # Check cache
        cache_key = self._compute_cache_key_for_analytical(
            partitioned_layer=partitioned_layer,
            chiplet_type=chiplet_type,
            batch_size=batch_size
        )
        
        # If cached, return the cached result
        if self.cache_manager.has_result(cache_key):
            print(f"🔄 Using cached analytical result for chiplet {chiplet_id}")
            return self.cache_manager.get_result(cache_key)
        
        # Run simulation
        result = self.analytical_backend.simulate(
            partitioned_layer=partitioned_layer,
            chiplet_id=chiplet_id,
            chiplet_params=chiplet_params,
            batch_size=batch_size
        )
        
        # Cache the result
        self.cache_manager.store_result(cache_key, result)
        
        return result
