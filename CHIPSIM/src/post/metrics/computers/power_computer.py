"""
Power metrics computer.

This module handles computation of power profiles over time, including compute
and communication power for each chiplet and the system as a whole.
"""

import json
import os
import pickle
import numpy as np
from typing import Any, Dict, Optional
from .base_computer import BaseMetricComputer


class PowerComputer(BaseMetricComputer):
    """
    Computes power profiles for chiplets and system over time.
    
    This computer calculates power consumption from:
    1. Compute operations (from chunk-level energy and latency data)
    2. Communication operations (from DSENT stats for routers and links)
    
    Attributes:
        dsent_stats_file_path: Path to DSENT statistics file (pickle or JSON)
        dsent_stats: Loaded DSENT statistics
        power_time_step_us: Time step used for power profiling
        power_time_points: Array of time points for power profiles
        chiplet_compute_power_over_time: Dict mapping chiplet ID to compute power array
        chiplet_activation_comm_power_over_time: Dict mapping chiplet ID to activation comm power array
        chiplet_weight_loading_power_over_time: Dict mapping chiplet ID to weight loading power array
        chiplet_communication_power_over_time: Dict mapping chiplet ID to combined comm power array
        chiplet_total_power_over_time: Dict mapping chiplet ID to total power array
        system_compute_power_over_time: Array of system-wide compute power
        system_activation_comm_power_over_time: Array of system-wide activation comm power
        system_weight_loading_power_over_time: Array of system-wide weight loading power
        system_communication_power_over_time: Array of system-wide communication power
        system_total_power_over_time: Array of system-wide total power
    """
    
    def __init__(self, retired_mapped_models: Dict[int, Any], global_time_us: float, 
                 num_chiplets: int, all_chiplets: set, dsent_stats_file_path: Optional[str] = None):
        """
        Initialize the power computer.
        
        Args:
            retired_mapped_models: Dictionary of completed mapped models
            global_time_us: Final global time of the simulation in microseconds
            num_chiplets: Total number of chiplets in the system
            all_chiplets: Set of all chiplet IDs used across all models
            dsent_stats_file_path: Path to DSENT statistics file (optional)
        """
        super().__init__(retired_mapped_models, global_time_us, num_chiplets, all_chiplets)
        self.dsent_stats_file_path = dsent_stats_file_path
        self.dsent_stats = None
        
        # Initialize power profile attributes
        self.power_time_step_us = None
        self.power_time_points = None
        self.chiplet_compute_power_over_time = None
        self.chiplet_activation_comm_power_over_time = None
        self.chiplet_weight_loading_power_over_time = None
        self.chiplet_communication_power_over_time = None
        self.chiplet_total_power_over_time = None
        self.system_compute_power_over_time = None
        self.system_activation_comm_power_over_time = None
        self.system_weight_loading_power_over_time = None
        self.system_communication_power_over_time = None
        self.system_total_power_over_time = None
    
    def _load_dsent_stats(self):
        """
        Loads DSENT stats from the file specified in self.dsent_stats_file_path.
        This method is idempotent; it will only load the file once.
        """
        # If stats are already loaded, do nothing.
        if self.dsent_stats is not None:
            return

        # If no file path is provided, initialize as an empty list and return.
        if not self.dsent_stats_file_path:
            print("Warning: DSENT stats file path not provided. Power metrics will be based on zero communication power.")
            self.dsent_stats = []
            return

        # Check if the file exists.
        if not os.path.exists(self.dsent_stats_file_path):
            print(f"Warning: DSENT stats file not found at '{self.dsent_stats_file_path}'. Power metrics will be based on zero communication power.")
            self.dsent_stats = []
            return
            
        # Load the stats from the DSENT stats file (pickle by default, legacy JSON fallback).
        loaded_stats = []
        try:
            if self.dsent_stats_file_path.endswith('.pkl'):
                with open(self.dsent_stats_file_path, 'rb') as f:
                    while True:
                        try:
                            loaded_stats.append(pickle.load(f))
                        except EOFError:
                            break
            else:
                with open(self.dsent_stats_file_path, 'r') as f:
                    for line in f:
                        try:
                            loaded_stats.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from a line in {self.dsent_stats_file_path}. Skipping line.")
            self.dsent_stats = loaded_stats
            print(f"Successfully loaded {len(self.dsent_stats)} entries from {self.dsent_stats_file_path}.")
        except Exception as e:
            print(f"Error: Failed to read DSENT stats file '{self.dsent_stats_file_path}': {e}")
            # Initialize as empty list on failure to prevent repeated attempts.
            self.dsent_stats = []

    def compute_power_profile(self, time_step_us=1.0):
        """
        Compute the power profile of the system over time, considering compute and communication.
        Uses chunk-level results for accurate per-chiplet power calculation.
        
        Args:
            time_step_us (float): The time step for the power profile in microseconds.
        """
        # Store the time step for later use in energy calculations
        self.power_time_step_us = time_step_us
        
        # Load DSENT stats if they haven't been loaded yet.
        self._load_dsent_stats()
        dsent_stats = self.dsent_stats

        # Determine the full set of chiplets in the system. All chiplets consume
        # communication power, even if idle.
        if self.num_chiplets > 0:
            all_system_chiplets = set(range(1, self.num_chiplets + 1))
        else:
            # Fallback to used chiplets if total number is not specified
            all_system_chiplets = self.all_chiplets
            if not all_system_chiplets:
                print("Warning: No chiplets to analyze (total configured number is zero and no chiplets were used). Power profile will be empty.")
            else:
                print(f"Warning: `num_chiplets` is not configured. Power metrics will only cover the {len(all_system_chiplets)} chiplets used in retired models.")

        if not all_system_chiplets:
            self.power_time_points = np.array([])
            self.chiplet_compute_power_over_time = {}
            self.chiplet_activation_comm_power_over_time = {}
            self.chiplet_weight_loading_power_over_time = {}
            self.chiplet_communication_power_over_time = {}
            self.chiplet_total_power_over_time = {}
            self.system_compute_power_over_time = np.array([])
            self.system_activation_comm_power_over_time = np.array([])
            self.system_weight_loading_power_over_time = np.array([])
            self.system_communication_power_over_time = np.array([])
            self.system_total_power_over_time = np.array([])
            return

        # --- Time points ---
        time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
        num_time_points = len(time_points)
        epsilon = 1e-9

        # --- Initialize power profiles ---
        chiplet_compute_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}
        chiplet_activation_comm_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}
        chiplet_weight_loading_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}

        # --- Process Compute Power using chunk results ---
        for model in self.retired_mapped_models.values():
            # Check if model has chunk_results
            if not hasattr(model, 'chunk_results'):
                continue
                
            for input_idx in range(model.num_inputs):
                if not all(hasattr(model, attr) for attr in ['layer_start_time_us']):
                    continue
                if not all(input_idx in getattr(model, attr) for attr in ['layer_start_time_us']):
                    continue

                for layer_idx in range(model.num_layers):
                    start_time = model.layer_start_time_us[input_idx].get(layer_idx)
                    
                    if start_time is None:
                        continue
                    
                    # Get chunk results for this layer
                    if layer_idx not in model.chunk_results:
                        continue
                    
                    # Process each chunk (chiplet) separately
                    for chiplet_id, (chunk_idx, chunk_result) in model.chunk_results[layer_idx].items():
                        # Get chunk-specific energy
                        chunk_energy_fj = chunk_result.get('energy_fj', 0)
                        if chunk_energy_fj <= 0:
                            continue
                        
                        # Get chunk-specific latency
                        chunk_latency_us = chunk_result.get('latency_us', 0)
                        if chunk_latency_us <= 0:
                            continue
                        
                        # Calculate chunk-specific power (convert fJ to J: 1e-15)
                        chunk_power_w = (chunk_energy_fj * 1e-15) / (chunk_latency_us * 1e-6)
                        
                        # The chunk is active from the layer start time for its own latency duration
                        activity_end_time = start_time + chunk_latency_us
                        
                        # Determine the range of time points this activity could affect
                        start_tp_idx = max(0, int(np.floor(start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil(activity_end_time / time_step_us)))
                        
                        # Add this chunk's power contribution to the chiplet's power profile
                        if chiplet_id in chiplet_compute_power_over_time:
                            for t_idx in range(start_tp_idx, end_tp_idx):
                                # Calculate the duration of overlap with this time bin
                                bin_start_time = time_points[t_idx]
                                bin_end_time = bin_start_time + time_step_us
                                
                                overlap_start = max(start_time, bin_start_time)
                                overlap_end = min(activity_end_time, bin_end_time)
                                overlap_duration = max(0, overlap_end - overlap_start)
                                
                                if overlap_duration > 0:
                                    # Prorate power: average power in this bin from this activity
                                    avg_power_in_bin = chunk_power_w * (overlap_duration / time_step_us)
                                    chiplet_compute_power_over_time[chiplet_id][t_idx] += avg_power_in_bin

        # --- Process Communication Power (Activation Communication and Weight Loading) ---
        if dsent_stats:
            sorted_dsent_stats = sorted(dsent_stats, key=lambda x: x['global_time_us'])
            for i, stats_entry in enumerate(sorted_dsent_stats):
                start_time = stats_entry['global_time_us']
                
                # Get the actual communication duration from the stats
                comm_runtime_us = 0.0
                
                # Extract runtime from the communication simulation results
                if 'latency' in stats_entry:
                    # Convert from seconds to microseconds if available
                    if 'total_runtime_s' in stats_entry['latency']:
                        comm_runtime_us = stats_entry['latency']['total_runtime_s'] * 1e6
                    elif 'total_runtime_us' in stats_entry['latency']:
                        comm_runtime_us = stats_entry['latency']['total_runtime_us']
                
                # If still no runtime, fall back to next simulation point (old behavior)
                if comm_runtime_us <= 0:
                    end_time = sorted_dsent_stats[i+1]['global_time_us'] if i + 1 < len(sorted_dsent_stats) else self.global_time_us
                else:
                    # Use actual communication duration
                    end_time = start_time + comm_runtime_us
                    # Ensure we don't exceed simulation time
                    end_time = min(end_time, self.global_time_us)
                
                start_tp_idx = max(0, int(np.floor(start_time / time_step_us)))
                end_tp_idx = min(num_time_points, int(np.ceil(end_time / time_step_us)))

                power_data = stats_entry.get('power', {})
                if not power_data:
                    continue

                # Determine if this is activation communication or weight loading based on phase type
                # For now, we'll distribute power to both activation communication and weight loading
                # In a more sophisticated implementation, we could determine the phase type from the stats
                is_activation_comm = True  # Default assumption
                is_weight_loading = True   # Default assumption
                
                # Router power - distribute to appropriate communication type
                for router_power_info in power_data.get('routers', []):
                    router_id = router_power_info.get('router_id')
                    chiplet_id = router_id + 1
                    
                    total_router_power = 0
                    if 'Total Power' in router_power_info:
                        power_dict = router_power_info.get('Total Power', {})
                        dynamic_power = power_dict.get('Dynamic power (W)', 0.0)
                        leakage_power = power_dict.get('Leakage power (W)', 0.0)
                        total_router_power = dynamic_power + leakage_power

                    if total_router_power > 0:
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            bin_start_time = time_points[t_idx]
                            bin_end_time = bin_start_time + time_step_us
                            overlap_start = max(start_time, bin_start_time)
                            overlap_end = min(end_time, bin_end_time)
                            overlap_duration = max(0, overlap_end - overlap_start)
                            if overlap_duration > 0:
                                avg_power = total_router_power * (overlap_duration / time_step_us)
                                # For now, distribute equally between activation communication and weight loading
                                # In a more sophisticated implementation, we could determine the actual phase type
                                if chiplet_id in chiplet_activation_comm_power_over_time:
                                    chiplet_activation_comm_power_over_time[chiplet_id][t_idx] += avg_power * 0.5
                                if chiplet_id in chiplet_weight_loading_power_over_time:
                                    chiplet_weight_loading_power_over_time[chiplet_id][t_idx] += avg_power * 0.5

                # Link power distribution - distribute to appropriate communication type
                total_link_power = power_data.get('totals', {}).get('link_power_W', 0)
                if total_link_power > 0:
                    chiplet_power_shares = {}
                    main_layer_info = stats_entry.get('main_layer_info')
                    traffic_dict = None
                    if main_layer_info:
                        model_idx, inp_idx, lay_idx = main_layer_info['model_idx'], main_layer_info['input_idx'], main_layer_info['layer_idx']
                        model = self.retired_mapped_models.get(model_idx)
                        if model and hasattr(model, 'layer_communication_sim_traffic'):
                            traffic_dict = model.layer_communication_sim_traffic.get(inp_idx, {}).get(lay_idx)

                    if traffic_dict:
                        chiplet_traffic = {cid: 0 for cid in all_system_chiplets}
                        total_traffic = 0
                        for (src, dst), data in traffic_dict.items():
                            amount = data.get('amount', 0)
                            if src in chiplet_traffic: chiplet_traffic[src] += amount
                            if dst in chiplet_traffic: chiplet_traffic[dst] += amount
                            total_traffic += amount
                        
                        if total_traffic > 0:
                            for cid in all_system_chiplets:
                                share = chiplet_traffic.get(cid, 0) / (2 * total_traffic)
                                power_share = total_link_power * share
                                chiplet_power_shares[cid] = power_share
                        else: # Fallback to even distribution
                            if all_system_chiplets:
                                power_per_chiplet = total_link_power / len(all_system_chiplets)
                                for cid in all_system_chiplets:
                                    chiplet_power_shares[cid] = power_per_chiplet
                    else: # Fallback to even distribution
                        if all_system_chiplets:
                            power_per_chiplet = total_link_power / len(all_system_chiplets)
                            for cid in all_system_chiplets:
                                chiplet_power_shares[cid] = power_per_chiplet
                    
                    # Apply prorated power for each chiplet's share
                    for cid, power_share in chiplet_power_shares.items():
                        if power_share > 0:
                            for t_idx in range(start_tp_idx, end_tp_idx):
                                bin_start_time = time_points[t_idx]
                                bin_end_time = bin_start_time + time_step_us
                                overlap_start = max(start_time, bin_start_time)
                                overlap_end = min(end_time, bin_end_time)
                                overlap_duration = max(0, overlap_end - overlap_start)
                                if overlap_duration > 0:
                                    avg_power = power_share * (overlap_duration / time_step_us)
                                    # For now, distribute equally between activation communication and weight loading
                                    # In a more sophisticated implementation, we could determine the actual phase type
                                    if cid in chiplet_activation_comm_power_over_time:
                                        chiplet_activation_comm_power_over_time[cid][t_idx] += avg_power * 0.5
                                    if cid in chiplet_weight_loading_power_over_time:
                                        chiplet_weight_loading_power_over_time[cid][t_idx] += avg_power * 0.5

        # --- Store results ---
        self.power_time_points = time_points
        self.chiplet_compute_power_over_time = chiplet_compute_power_over_time
        self.chiplet_activation_comm_power_over_time = chiplet_activation_comm_power_over_time
        self.chiplet_weight_loading_power_over_time = chiplet_weight_loading_power_over_time
        
        self.chiplet_total_power_over_time = {
            cid: chiplet_compute_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_activation_comm_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_weight_loading_power_over_time.get(cid, np.zeros(num_time_points))
            for cid in all_system_chiplets
        }
        
        self.system_compute_power_over_time = np.sum(list(chiplet_compute_power_over_time.values()), axis=0) if chiplet_compute_power_over_time else np.zeros(num_time_points)
        self.system_activation_comm_power_over_time = np.sum(list(chiplet_activation_comm_power_over_time.values()), axis=0) if chiplet_activation_comm_power_over_time else np.zeros(num_time_points)
        self.system_weight_loading_power_over_time = np.sum(list(chiplet_weight_loading_power_over_time.values()), axis=0) if chiplet_weight_loading_power_over_time else np.zeros(num_time_points)
        self.system_total_power_over_time = self.system_compute_power_over_time + self.system_activation_comm_power_over_time + self.system_weight_loading_power_over_time
        
        # Create combined communication power attributes for compatibility with plotter
        self.chiplet_communication_power_over_time = {
            cid: chiplet_activation_comm_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_weight_loading_power_over_time.get(cid, np.zeros(num_time_points))
            for cid in all_system_chiplets
        }
        self.system_communication_power_over_time = self.system_activation_comm_power_over_time + self.system_weight_loading_power_over_time
