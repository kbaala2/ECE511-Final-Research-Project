"""
Model summary metrics computer.

This module handles computation of per-model summary metrics including
latency, energy, and traffic statistics.
"""

from typing import Any, Dict
from .base_computer import BaseMetricComputer


class ModelSummaryComputer(BaseMetricComputer):
    """
    Computes summary metrics for each model.
    
    Calculates aggregate statistics for each model including:
    - Maximum single input latencies (compute, communication, total)
    - Summed latencies across all inputs
    - Total energy consumption
    - Communication traffic statistics
    - Overall model duration
    
    Attributes:
        model_summary_metrics: Dictionary mapping model_idx to summary metrics
    """
    
    def __init__(self, retired_mapped_models: Dict[int, Any], global_time_us: float, 
                 num_chiplets: int, all_chiplets: set):
        """
        Initialize the model summary computer.
        
        Args:
            retired_mapped_models: Dictionary of completed mapped models
            global_time_us: Final global time of the simulation in microseconds
            num_chiplets: Total number of chiplets in the system
            all_chiplets: Set of all chiplet IDs used across all models
        """
        super().__init__(retired_mapped_models, global_time_us, num_chiplets, all_chiplets)
        self.model_summary_metrics = None
    
    def compute_model_summary_metrics(self):
        """
        Computes summary metrics for each model (latency, energy, traffic).
        Stores results in the `self.model_summary_metrics` attribute.
        Handles both pipelined and non-pipelined scenarios.
        
        Returns:
            dict: A dictionary containing summary metrics for each model.
                  Key: model_idx
                  Value: dict of metrics
        """
        
        # Assert that retired_mapped_models exists and is not empty
        assert self.retired_mapped_models, "ERROR: No retired models found. Metrics cannot be computed."
        
        self.model_summary_metrics = {}
        
        for model_idx, mapped_model in self.retired_mapped_models.items():
            metrics = {}
            
            # --- Basic attribute validation with detailed assertions ---
            model_name = getattr(mapped_model, 'model_name', f"Model {model_idx}")
            
            # Required attributes for metric computation
            assert hasattr(mapped_model, 'num_inputs'), f"ERROR: Model {model_idx} ({model_name}) missing 'num_inputs' attribute"
            assert mapped_model.num_inputs > 0, f"ERROR: Model {model_idx} ({model_name}) has invalid num_inputs: {mapped_model.num_inputs}"

            assert hasattr(mapped_model, 'layer_compute_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_compute_latency' attribute"
            assert hasattr(mapped_model, 'layer_activation_comm_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_activation_comm_latency' attribute"
            assert hasattr(mapped_model, 'layer_weight_loading_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_weight_loading_latency' attribute"
            
            # Verify data integrity - at least one input should have latency data
            valid_compute_data = False
            valid_activation_comm_data = False
            valid_weight_loading_data = False
            
            for input_idx in range(mapped_model.num_inputs):
                if input_idx in mapped_model.layer_compute_latency:
                    valid_compute_data = True
                if input_idx in mapped_model.layer_activation_comm_latency:
                    valid_activation_comm_data = True
                if input_idx in mapped_model.layer_weight_loading_latency:
                    valid_weight_loading_data = True
            
            assert valid_compute_data, f"ERROR: Model {model_idx} ({model_name}) has no valid compute latency data for any input"
            assert valid_activation_comm_data, f"ERROR: Model {model_idx} ({model_name}) has no valid activation communication latency data for any input"
            assert valid_weight_loading_data, f"ERROR: Model {model_idx} ({model_name}) has no valid weight loading latency data for any input"

            # Initialize metrics
            max_single_input_total_latency = 0.0
            max_single_input_compute_latency = 0.0
            max_single_input_activation_comm_latency = 0.0
            max_single_input_weight_loading_latency = 0.0

            sum_compute_latency_all_inputs = 0.0
            sum_activation_comm_latency_all_inputs = 0.0
            sum_weight_loading_latency_all_inputs = 0.0
            total_compute_energy_all_inputs = 0.0
            total_activation_comm_traffic = 0
            total_weight_loading_traffic = 0

            num_valid_inputs = 0
            all_inputs_latency_data = []

            # Calculate totals and maximums across all inputs
            for input_idx in range(mapped_model.num_inputs):
                input_compute_latency = 0.0
                input_activation_comm_latency = 0.0
                input_weight_loading_latency = 0.0
                
                if input_idx in mapped_model.layer_compute_latency:
                    for layer_idx, latency in mapped_model.layer_compute_latency[input_idx].items():
                        if latency > 0:
                            input_compute_latency += latency
                
                if input_idx in mapped_model.layer_activation_comm_latency:
                    for layer_idx, latency in mapped_model.layer_activation_comm_latency[input_idx].items():
                        if latency > 0:
                            input_activation_comm_latency += latency
                
                if input_idx in mapped_model.layer_weight_loading_latency:
                    for layer_idx, latency in mapped_model.layer_weight_loading_latency[input_idx].items():
                        if latency > 0:
                            input_weight_loading_latency += latency
                
                # Only count this input if it has any positive latency values
                if input_compute_latency > 0 or input_activation_comm_latency > 0 or input_weight_loading_latency > 0:
                    num_valid_inputs += 1
                    input_total_latency = input_compute_latency + input_activation_comm_latency + input_weight_loading_latency
                    all_inputs_latency_data.append((input_idx, input_compute_latency, input_activation_comm_latency, input_weight_loading_latency, input_total_latency))
                    
                    # Track maximum single input latencies
                    max_single_input_total_latency = max(max_single_input_total_latency, input_total_latency)
                    max_single_input_compute_latency = max(max_single_input_compute_latency, input_compute_latency)
                    max_single_input_activation_comm_latency = max(max_single_input_activation_comm_latency, input_activation_comm_latency)
                    max_single_input_weight_loading_latency = max(max_single_input_weight_loading_latency, input_weight_loading_latency)
                    
                    # Accumulate sums across all inputs
                    sum_compute_latency_all_inputs += input_compute_latency
                    sum_activation_comm_latency_all_inputs += input_activation_comm_latency
                    sum_weight_loading_latency_all_inputs += input_weight_loading_latency
                
                # Accumulate energy (separate from latency check)
                if hasattr(mapped_model, 'layer_compute_energy') and input_idx in mapped_model.layer_compute_energy:
                    total_compute_energy_all_inputs += sum(energy for energy in mapped_model.layer_compute_energy[input_idx].values() if energy >= 0)

            # Assert that at least one input has valid latency data
            assert num_valid_inputs > 0, f"ERROR: Model {model_idx} ({model_name}) has no inputs with valid latency data"

            # Calculate total model latency (summed work)
            total_model_latency_summed_work = sum_compute_latency_all_inputs + sum_activation_comm_latency_all_inputs + sum_weight_loading_latency_all_inputs
            assert total_model_latency_summed_work > 0, f"ERROR: Model {model_idx} ({model_name}) has zero total latency"

            # Calculate total activation communication traffic safely
            if hasattr(mapped_model, 'layer_activation_comm_traffic'):
                for layer_idx, traffic_data in mapped_model.layer_activation_comm_traffic.items():
                    if isinstance(traffic_data, dict):
                        for src_chiplet, dest_map in traffic_data.items():
                            if isinstance(dest_map, dict):
                                total_activation_comm_traffic += sum(v for v in dest_map.values() if isinstance(v, (int, float)) and v >= 0)
                    elif isinstance(traffic_data, (int, float)) and traffic_data >= 0:
                        total_activation_comm_traffic += traffic_data

            # Calculate total weight loading traffic safely
            if hasattr(mapped_model, 'layer_weight_loading_traffic'):
                for layer_idx, traffic_data in mapped_model.layer_weight_loading_traffic.items():
                    if isinstance(traffic_data, dict):
                        for src_chiplet, dest_map in traffic_data.items():
                            if isinstance(dest_map, dict):
                                total_weight_loading_traffic += sum(v for v in dest_map.values() if isinstance(v, (int, float)) and v >= 0)
                    elif isinstance(traffic_data, (int, float)) and traffic_data >= 0:
                        total_weight_loading_traffic += traffic_data

            # Get overall model start/end times
            model_start_time = getattr(mapped_model, 'model_start_time_us', -1.0)
            model_completion_time = getattr(mapped_model, 'model_completion_time_us', -1.0)
            
            # Always ensure a valid duration is calculated
            if model_start_time >= 0 and model_completion_time >= model_start_time:
                overall_model_duration = model_completion_time - model_start_time
            elif hasattr(mapped_model, 'input_completion_time_us') and model_start_time >= 0:
                # Try to use input completion times as a fallback
                valid_completion_times = [t for t in mapped_model.input_completion_time_us if t >= 0]
                if valid_completion_times:
                    latest_completion = max(valid_completion_times)
                    overall_model_duration = latest_completion - model_start_time
                    print(f"INFO: Model {model_idx} ({model_name}): Using max input completion time for duration: {overall_model_duration:.2f} μs")
                else:
                    overall_model_duration = -1.0
            else:
                overall_model_duration = -1.0
                
            # At this point, metrics should be valid because of our assertions
            metrics['max_single_input_total_latency'] = max_single_input_total_latency
            metrics['max_single_input_compute_latency'] = max_single_input_compute_latency
            metrics['max_single_input_activation_comm_latency'] = max_single_input_activation_comm_latency
            metrics['max_single_input_weight_loading_latency'] = max_single_input_weight_loading_latency
            metrics['sum_compute_latency_all_inputs'] = sum_compute_latency_all_inputs
            metrics['sum_activation_comm_latency_all_inputs'] = sum_activation_comm_latency_all_inputs
            metrics['sum_weight_loading_latency_all_inputs'] = sum_weight_loading_latency_all_inputs
            metrics['total_compute_energy_all_inputs'] = total_compute_energy_all_inputs
            metrics['total_model_latency_summed_work'] = total_model_latency_summed_work
            metrics['total_activation_comm_traffic'] = total_activation_comm_traffic
            metrics['total_weight_loading_traffic'] = total_weight_loading_traffic
            metrics['overall_model_duration'] = overall_model_duration
            
            # Store the metrics in our dictionary
            self.model_summary_metrics[model_idx] = metrics
            print(f"Successfully computed metrics for Model {model_idx} ({model_name})")

        # Verify metrics were computed for all models
        assert len(self.model_summary_metrics) == len(self.retired_mapped_models), \
               f"ERROR: Metrics computed for only {len(self.model_summary_metrics)} models out of {len(self.retired_mapped_models)}"

        print(f"Computed metrics for all {len(self.model_summary_metrics)} models")
        return self.model_summary_metrics
