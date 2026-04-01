#!/usr/bin/env python3
# class_MetricComputer.py

import json
import os
import pickle

import numpy as np

# Import specialized metric computers (Phase 2, 3 & 4 refactoring)
from .computers.utilization_computer import UtilizationComputer
from .computers.power_computer import PowerComputer
from .computers.energy_computer import EnergyComputer
from .computers.model_summary_computer import ModelSummaryComputer
from .computers.comparison_computer import ComparisonComputer

class MetricComputer:
    """
    Computes performance metrics based on simulation results.
    
    Args:
        retired_mapped_models (dict): Dictionary of completed mapped models
        global_time_us (float): Final global time of the simulation in microseconds
    """
    def __init__(self, 
                 retired_mapped_models, 
                 global_time_us,
                 num_chiplets,
                 dsent_stats_file_path=None):
        
        self.retired_mapped_models = retired_mapped_models
        self.global_time_us = global_time_us
        self.total_simulation_time_us = global_time_us  # Alias for cross-simulation compatibility
        self.num_chiplets = num_chiplets
        self.dsent_stats_file_path = dsent_stats_file_path
        self.dsent_stats = None
        
        # Extract all chiplets used across all models
        self.all_chiplets = set()
        for model in retired_mapped_models.values():
            self.all_chiplets.update(model.get_chiplets_used())
        
        # Initialize specialized computer modules (Phase 2, 3 & 4 refactoring)
        self.utilization_computer = UtilizationComputer(
            retired_mapped_models=self.retired_mapped_models,
            global_time_us=self.global_time_us,
            num_chiplets=self.num_chiplets,
            all_chiplets=self.all_chiplets
        )
        self.power_computer = PowerComputer(
            retired_mapped_models=self.retired_mapped_models,
            global_time_us=self.global_time_us,
            num_chiplets=self.num_chiplets,
            all_chiplets=self.all_chiplets,
            dsent_stats_file_path=self.dsent_stats_file_path
        )
        self.energy_computer = EnergyComputer(
            retired_mapped_models=self.retired_mapped_models,
            global_time_us=self.global_time_us,
            num_chiplets=self.num_chiplets,
            all_chiplets=self.all_chiplets
        )
        self.model_summary_computer = ModelSummaryComputer(
            retired_mapped_models=self.retired_mapped_models,
            global_time_us=self.global_time_us,
            num_chiplets=self.num_chiplets,
            all_chiplets=self.all_chiplets
        )
        self.comparison_computer = ComparisonComputer(
            retired_mapped_models=self.retired_mapped_models,
            global_time_us=self.global_time_us,
            num_chiplets=self.num_chiplets,
            all_chiplets=self.all_chiplets
        )

        # Initialize attributes to store computed metrics

        # ---- Utilization Metrics ----
        # Method 1: Total compute time based utilization
        self.per_chiplet_compute_time_utilization_pct = None  # Renamed from chiplet_utilization
        self.chiplet_total_compute_time_us = None  # Renamed from chiplet_compute_time
        self.mean_compute_time_chiplet_utilization_pct = None # Was compute_time_avg_utilization_pct

        # New attributes for activation communication time based utilization (Method 1 style)
        self.per_chiplet_activation_comm_time_utilization_pct = None
        self.chiplet_total_activation_comm_time_us = None
        self.mean_activation_comm_time_chiplet_utilization_pct = None

        # New attributes for weight loading communication time based utilization (Method 1 style)
        self.per_chiplet_weight_loading_time_utilization_pct = None
        self.chiplet_total_weight_loading_time_us = None
        self.mean_weight_loading_time_chiplet_utilization_pct = None

        # New attributes for combined (compute + activation_comm + weight_loading) time based utilization (Method 1 style)
        self.per_chiplet_combined_time_utilization_pct = None
        self.chiplet_total_combined_busy_time_us = None # Sum of (compute_latency + activation_comm_latency + weight_loading_latency) for tasks
        self.mean_combined_time_chiplet_utilization_pct = None
        
        # Method 2: Time-step based activity tracking
        # Compute Utilization
        self.utilization_time_points = None # Shared across compute, communication, and combined
        self.chiplet_compute_activity_at_time_step = None # {time_point: {chiplet_id: activity (0 or 1), ...}}
        self.chiplet_compute_activity_over_time = None
        self.system_compute_utilization_over_time_pct = None
        
        # Activation Communication Utilization
        self.chiplet_activation_comm_activity_at_time_step = None
        self.chiplet_activation_comm_activity_over_time = None
        self.system_activation_comm_utilization_over_time_pct = None
        
        # Weight Loading Communication Utilization
        self.chiplet_weight_loading_activity_at_time_step = None
        self.chiplet_weight_loading_activity_over_time = None
        self.system_weight_loading_utilization_over_time_pct = None
        
        # Combined Utilization (Compute + Activation Communication + Weight Loading)
        self.chiplet_combined_activity_at_time_step = None
        self.chiplet_combined_activity_over_time = None
        self.system_combined_utilization_over_time_pct = None
        
        # Original detailed activity tracking (renamed) - this was `self.chiplet_activity_at_time_step` before
        # This specific attribute might be redundant if the new ones cover its purpose,
        # but I'm keeping its structure as per the original code's intent for "Method 3".
        # It was described as: self.chiplet_activity_at_time_step = None # {time_point: {chiplet_id: activity (0 or 1), ...}}
        # Let's rename it to avoid collision and clarify its original purpose if it's still needed.
        self.detailed_general_activity_at_time_step = None 
        
        self.model_summary_metrics = None
        
        # ---- Power Metrics ----
        self.power_time_points = None
        self.power_time_step_us = None
        self.chiplet_compute_power_over_time = None
        self.chiplet_activation_comm_power_over_time = None
        self.chiplet_weight_loading_power_over_time = None
        self.chiplet_total_power_over_time = None
        self.system_compute_power_over_time = None
        self.system_activation_comm_power_over_time = None
        self.system_weight_loading_power_over_time = None
        self.system_total_power_over_time = None
        
        # ---- Energy Metrics ----
        self.chiplet_compute_energy_uj = None
        self.chiplet_activation_comm_energy_uj = None
        self.chiplet_weight_loading_energy_uj = None
        # Combined communication energy (activation + weight loading)
        # These are populated in compute_energy_metrics()
        self.chiplet_communication_energy_uj = None
        self.chiplet_total_energy_uj = None
        self.system_compute_energy_uj = None
        self.system_activation_comm_energy_uj = None
        self.system_weight_loading_energy_uj = None
        # Combined communication energy (activation + weight loading)
        self.system_communication_energy_uj = None
        self.system_total_energy_uj = None
        
        self.approach_comparison_metrics = {
            "avg_time_per_input_per_model_type_main_sim": None,
            "avg_time_per_input_per_model_type_comparison": None
        }

    def compute_avg_system_utilization(self):
        """
        Computes the system compute utilization per chiplet based on total compute time.
        
        This method now delegates to UtilizationComputer for the actual computation
        (Phase 2 refactoring). Results are copied back to MetricComputer attributes
        for backward compatibility.

        Returns:
            None. Results are stored in instance attributes.
        """
        # Delegate to UtilizationComputer
        self.utilization_computer.compute_avg_system_utilization()
        
        # Copy results back to self for backward compatibility
        self.chiplet_total_compute_time_us = self.utilization_computer.chiplet_total_compute_time_us
        self.chiplet_total_activation_comm_time_us = self.utilization_computer.chiplet_total_activation_comm_time_us
        self.chiplet_total_weight_loading_time_us = self.utilization_computer.chiplet_total_weight_loading_time_us
        self.chiplet_total_combined_busy_time_us = self.utilization_computer.chiplet_total_combined_busy_time_us
        
        self.per_chiplet_compute_time_utilization_pct = self.utilization_computer.per_chiplet_compute_time_utilization_pct
        self.per_chiplet_activation_comm_time_utilization_pct = self.utilization_computer.per_chiplet_activation_comm_time_utilization_pct
        self.per_chiplet_weight_loading_time_utilization_pct = self.utilization_computer.per_chiplet_weight_loading_time_utilization_pct
        self.per_chiplet_combined_time_utilization_pct = self.utilization_computer.per_chiplet_combined_time_utilization_pct
        
        self.mean_compute_time_chiplet_utilization_pct = self.utilization_computer.mean_compute_time_chiplet_utilization_pct
        self.mean_activation_comm_time_chiplet_utilization_pct = self.utilization_computer.mean_activation_comm_time_chiplet_utilization_pct
        self.mean_weight_loading_time_chiplet_utilization_pct = self.utilization_computer.mean_weight_loading_time_chiplet_utilization_pct
        self.mean_combined_time_chiplet_utilization_pct = self.utilization_computer.mean_combined_time_chiplet_utilization_pct
        self.mean_communication_time_chiplet_utilization_pct = self.utilization_computer.mean_communication_time_chiplet_utilization_pct
        
        return

    def compute_utilization_over_time(self, time_step_us=1.0):
        """
        Computes chiplet COMPUTE, COMMUNICATION, and COMBINED activity at each discrete time step
        and derives utilization metrics, storing the results in instance attributes.

        This method now delegates to UtilizationComputer for the actual computation
        (Phase 2 refactoring). Results are copied back to MetricComputer attributes
        for backward compatibility.

        Args:
            time_step_us (float): Time step in microseconds for sampling utilization.

        Returns:
            None
        """
        # Delegate to UtilizationComputer
        self.utilization_computer.compute_utilization_over_time(time_step_us)
        
        # Copy results back to self for backward compatibility
        self.utilization_time_points = self.utilization_computer.utilization_time_points
        
        self.chiplet_compute_activity_at_time_step = self.utilization_computer.chiplet_compute_activity_at_time_step
        self.chiplet_compute_activity_over_time = self.utilization_computer.chiplet_compute_activity_over_time
        self.system_compute_utilization_over_time_pct = self.utilization_computer.system_compute_utilization_over_time_pct
        
        self.chiplet_activation_comm_activity_at_time_step = self.utilization_computer.chiplet_activation_comm_activity_at_time_step
        self.chiplet_activation_comm_activity_over_time = self.utilization_computer.chiplet_activation_comm_activity_over_time
        self.system_activation_comm_utilization_over_time_pct = self.utilization_computer.system_activation_comm_utilization_over_time_pct
        
        self.chiplet_weight_loading_activity_at_time_step = self.utilization_computer.chiplet_weight_loading_activity_at_time_step
        self.chiplet_weight_loading_activity_over_time = self.utilization_computer.chiplet_weight_loading_activity_over_time
        self.system_weight_loading_utilization_over_time_pct = self.utilization_computer.system_weight_loading_utilization_over_time_pct
        
        self.chiplet_combined_activity_at_time_step = self.utilization_computer.chiplet_combined_activity_at_time_step
        self.chiplet_combined_activity_over_time = self.utilization_computer.chiplet_combined_activity_over_time
        self.system_combined_utilization_over_time_pct = self.utilization_computer.system_combined_utilization_over_time_pct
        
        return

    def compute_model_summary_metrics(self):
        """
        Computes summary metrics for each model (latency, energy, traffic).
        
        This method now delegates to ModelSummaryComputer (Phase 4 refactoring).
        Results are copied back to MetricComputer attributes for backward compatibility.
        
        Returns:
            dict: A dictionary containing summary metrics for each model.
        """
        # Delegate to ModelSummaryComputer
        result = self.model_summary_computer.compute_model_summary_metrics()
        
        # Copy results back to self for backward compatibility
        self.model_summary_metrics = self.model_summary_computer.model_summary_metrics
        
        return result

    def compute_approach_comparison_metrics(self, individual_results=None, empty_system_individual_model_results=None):
        """
        Computes model execution times and averages, including separate compute and communication metrics.
        
        This method now delegates to ComparisonComputer (Phase 4 refactoring).
        Results are copied back to MetricComputer attributes for backward compatibility.
        
        Args:
            individual_results: Results from individual layer communication simulation
            empty_system_individual_model_results: Pre-calculated results for model types in empty system
        
        Returns:
            dict: The approach_comparison_metrics dictionary
        """
        # Delegate to ComparisonComputer
        result = self.comparison_computer.compute_approach_comparison_metrics(
            individual_results=individual_results,
            empty_system_individual_model_results=empty_system_individual_model_results
        )
        
        # Copy results back to self for backward compatibility
        self.approach_comparison_metrics = self.comparison_computer.approach_comparison_metrics
        
        return result

    def _load_dsent_stats(self):
        """
        Loads DSENT stats from the file specified in self.dsent_stats_file_path.
        
        This method now delegates to PowerComputer (Phase 3 refactoring).
        Results are copied back to MetricComputer attributes for backward compatibility.
        """
        # Delegate to PowerComputer
        self.power_computer._load_dsent_stats()
        
        # Copy result back for backward compatibility
        self.dsent_stats = self.power_computer.dsent_stats

    def compute_power_profile(self, time_step_us=1.0):
        """
        Compute the power profile of the system over time, considering compute and communication.
        Uses chunk-level results for accurate per-chiplet power calculation.
        
        This method now delegates to PowerComputer (Phase 3 refactoring).
        Results are copied back to MetricComputer attributes for backward compatibility.
        
        Args:
            time_step_us (float): The time step for the power profile in microseconds.
        """
        # Delegate to PowerComputer
        self.power_computer.compute_power_profile(time_step_us)
        
        # Copy results back to self for backward compatibility
        self.power_time_step_us = self.power_computer.power_time_step_us
        self.power_time_points = self.power_computer.power_time_points
        self.chiplet_compute_power_over_time = self.power_computer.chiplet_compute_power_over_time
        self.chiplet_activation_comm_power_over_time = self.power_computer.chiplet_activation_comm_power_over_time
        self.chiplet_weight_loading_power_over_time = self.power_computer.chiplet_weight_loading_power_over_time
        self.chiplet_communication_power_over_time = self.power_computer.chiplet_communication_power_over_time
        self.chiplet_total_power_over_time = self.power_computer.chiplet_total_power_over_time
        self.system_compute_power_over_time = self.power_computer.system_compute_power_over_time
        self.system_activation_comm_power_over_time = self.power_computer.system_activation_comm_power_over_time
        self.system_weight_loading_power_over_time = self.power_computer.system_weight_loading_power_over_time
        self.system_communication_power_over_time = self.power_computer.system_communication_power_over_time
        self.system_total_power_over_time = self.power_computer.system_total_power_over_time

    def compute_energy_metrics(self):
        """
        Computes energy consumption metrics based on the computed power profiles.
        This method should be called *after* compute_power_profile.

        This method now delegates to EnergyComputer (Phase 3 refactoring).
        Results are copied back to MetricComputer attributes for backward compatibility.

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
        # Delegate to EnergyComputer
        self.energy_computer.compute_energy_metrics(self.power_computer)
        
        # Copy results back to self for backward compatibility
        self.chiplet_compute_energy_uj = self.energy_computer.chiplet_compute_energy_uj
        self.chiplet_activation_comm_energy_uj = self.energy_computer.chiplet_activation_comm_energy_uj
        self.chiplet_weight_loading_energy_uj = self.energy_computer.chiplet_weight_loading_energy_uj
        self.chiplet_communication_energy_uj = self.energy_computer.chiplet_communication_energy_uj
        self.chiplet_total_energy_uj = self.energy_computer.chiplet_total_energy_uj
        self.system_compute_energy_uj = self.energy_computer.system_compute_energy_uj
        self.system_activation_comm_energy_uj = self.energy_computer.system_activation_comm_energy_uj
        self.system_weight_loading_energy_uj = self.energy_computer.system_weight_loading_energy_uj
        self.system_communication_energy_uj = self.energy_computer.system_communication_energy_uj
        self.system_total_energy_uj = self.energy_computer.system_total_energy_uj