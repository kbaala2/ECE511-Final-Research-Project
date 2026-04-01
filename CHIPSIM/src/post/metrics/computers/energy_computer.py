"""
Energy metrics computer.

This module handles computation of energy metrics from power profiles.
Energy is calculated by integrating power over time.
"""

import numpy as np
from typing import Any, Dict, Optional
from .base_computer import BaseMetricComputer


class EnergyComputer(BaseMetricComputer):
    """
    Computes energy consumption metrics from power profiles.
    
    This computer calculates total energy consumed for compute, communication,
    and their sum, both on a per-chiplet basis and for the entire system.
    
    The energy is calculated by integrating the power over time, approximated by
    summing the power at each time step multiplied by the time step duration.
    
    Note: This computer depends on power profiles being computed first by PowerComputer.
    
    Attributes:
        chiplet_compute_energy_uj: Dict mapping chiplet ID to compute energy (microJoules)
        chiplet_activation_comm_energy_uj: Dict mapping chiplet ID to activation comm energy
        chiplet_weight_loading_energy_uj: Dict mapping chiplet ID to weight loading energy
        chiplet_communication_energy_uj: Dict mapping chiplet ID to combined comm energy
        chiplet_total_energy_uj: Dict mapping chiplet ID to total energy
        system_compute_energy_uj: Total system compute energy (microJoules)
        system_activation_comm_energy_uj: Total system activation comm energy
        system_weight_loading_energy_uj: Total system weight loading energy
        system_communication_energy_uj: Total system communication energy
        system_total_energy_uj: Total system energy
    """
    
    def __init__(self, retired_mapped_models: Dict[int, Any], global_time_us: float, 
                 num_chiplets: int, all_chiplets: set):
        """
        Initialize the energy computer.
        
        Args:
            retired_mapped_models: Dictionary of completed mapped models
            global_time_us: Final global time of the simulation in microseconds
            num_chiplets: Total number of chiplets in the system
            all_chiplets: Set of all chiplet IDs used across all models
        """
        super().__init__(retired_mapped_models, global_time_us, num_chiplets, all_chiplets)
        
        # Initialize energy metric attributes
        self.chiplet_compute_energy_uj = None
        self.chiplet_activation_comm_energy_uj = None
        self.chiplet_weight_loading_energy_uj = None
        self.chiplet_communication_energy_uj = None
        self.chiplet_total_energy_uj = None
        self.system_compute_energy_uj = None
        self.system_activation_comm_energy_uj = None
        self.system_weight_loading_energy_uj = None
        self.system_communication_energy_uj = None
        self.system_total_energy_uj = None
    
    def compute_energy_metrics(self, power_computer):
        """
        Computes energy consumption metrics based on the computed power profiles.
        This method should be called *after* PowerComputer.compute_power_profile().

        It calculates the total energy consumed for compute, communication, and their
        sum, both on a per-chiplet basis and for the entire system.

        The energy is calculated by integrating the power over time, approximated by
        summing the power at each time step multiplied by the time step duration.

        Args:
            power_computer: PowerComputer instance with computed power profiles

        Updates the following instance attributes:
        - self.chiplet_compute_energy_uj
        - self.chiplet_activation_comm_energy_uj
        - self.chiplet_weight_loading_energy_uj
        - self.chiplet_communication_energy_uj
        - self.chiplet_total_energy_uj
        - self.system_compute_energy_uj
        - self.system_activation_comm_energy_uj
        - self.system_weight_loading_energy_uj
        - self.system_communication_energy_uj
        - self.system_total_energy_uj
        """
        # --- Pre-computation Checks ---
        if power_computer.power_time_points is None or power_computer.power_time_step_us is None:
            print("Warning: Power profiles must be computed before energy metrics can be calculated. Call `compute_power_profile` first.")
            return

        if power_computer.power_time_step_us <= 0:
            print(f"Warning: Invalid power time step ({power_computer.power_time_step_us} us). Cannot compute energy metrics.")
            return

        # Energy in microJoules (uJ) is Power (W) * time (us).
        time_step_us = power_computer.power_time_step_us

        # --- Per-Chiplet Energy Calculation ---
        self.chiplet_compute_energy_uj = {}
        if power_computer.chiplet_compute_power_over_time:
            for cid, power_profile in power_computer.chiplet_compute_power_over_time.items():
                self.chiplet_compute_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # Calculate activation communication energy
        self.chiplet_activation_comm_energy_uj = {}
        if power_computer.chiplet_activation_comm_power_over_time:
            for cid, power_profile in power_computer.chiplet_activation_comm_power_over_time.items():
                self.chiplet_activation_comm_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # Calculate weight loading energy
        self.chiplet_weight_loading_energy_uj = {}
        if power_computer.chiplet_weight_loading_power_over_time:
            for cid, power_profile in power_computer.chiplet_weight_loading_power_over_time.items():
                self.chiplet_weight_loading_energy_uj[cid] = np.sum(power_profile) * time_step_us
        
        # Combined communication energy per chiplet (activation + weight loading)
        self.chiplet_communication_energy_uj = {}
        # Use union of keys from both dicts to be robust
        all_chiplet_ids = set(self.chiplet_activation_comm_energy_uj.keys()) | set(self.chiplet_weight_loading_energy_uj.keys())
        for cid in all_chiplet_ids:
            act_e = self.chiplet_activation_comm_energy_uj.get(cid, 0.0)
            wl_e = self.chiplet_weight_loading_energy_uj.get(cid, 0.0)
            self.chiplet_communication_energy_uj[cid] = act_e + wl_e

        self.chiplet_total_energy_uj = {}
        if power_computer.chiplet_total_power_over_time:
            for cid, power_profile in power_computer.chiplet_total_power_over_time.items():
                self.chiplet_total_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # --- System-wide Energy Calculation ---
        if power_computer.system_compute_power_over_time is not None:
            self.system_compute_energy_uj = np.sum(power_computer.system_compute_power_over_time) * time_step_us
        else:
            self.system_compute_energy_uj = 0.0

        if power_computer.system_activation_comm_power_over_time is not None:
            self.system_activation_comm_energy_uj = np.sum(power_computer.system_activation_comm_power_over_time) * time_step_us
        else:
            self.system_activation_comm_energy_uj = 0.0

        if power_computer.system_weight_loading_power_over_time is not None:
            self.system_weight_loading_energy_uj = np.sum(power_computer.system_weight_loading_power_over_time) * time_step_us
        else:
            self.system_weight_loading_energy_uj = 0.0

        # Combined communication energy (activation + weight loading)
        self.system_communication_energy_uj = (
            (self.system_activation_comm_energy_uj or 0.0) +
            (self.system_weight_loading_energy_uj or 0.0)
        )

        if power_computer.system_total_power_over_time is not None:
            self.system_total_energy_uj = np.sum(power_computer.system_total_power_over_time) * time_step_us
        else:
            self.system_total_energy_uj = 0.0
            
        # As a check, system total should be sum of compute, activation communication, and weight loading
        # Allow for small floating point discrepancies
        if (self.system_compute_energy_uj is not None and 
            self.system_activation_comm_energy_uj is not None and 
            self.system_weight_loading_energy_uj is not None):
            assert np.isclose(
                self.system_total_energy_uj, 
                self.system_compute_energy_uj + 
                self.system_activation_comm_energy_uj + 
                self.system_weight_loading_energy_uj
            ), "System total energy does not match the sum of compute and communication energy components."
