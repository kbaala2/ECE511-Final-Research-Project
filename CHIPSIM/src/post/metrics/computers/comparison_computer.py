"""
Approach comparison metrics computer.

This module handles computation of comparison metrics across different
simulation approaches (main simulation, individual model, empty system).
"""

from typing import Any, Dict, Optional
from .base_computer import BaseMetricComputer


class ComparisonComputer(BaseMetricComputer):
    """
    Computes comparison metrics across different simulation approaches.
    
    Compares model execution times and averages across three sources:
    1. Main simulation (self.retired_mapped_models)
    2. Individual model (main sim compute + communication from individual_results)
    3. Empty system model (from empty_system_individual_model_results)
    
    Also computes separate average compute and communication times per model type.
    
    Attributes:
        approach_comparison_metrics: Dictionary containing comparison data
    """
    
    def __init__(self, retired_mapped_models: Dict[int, Any], global_time_us: float, 
                 num_chiplets: int, all_chiplets: set):
        """
        Initialize the comparison computer.
        
        Args:
            retired_mapped_models: Dictionary of completed mapped models
            global_time_us: Final global time of the simulation in microseconds
            num_chiplets: Total number of chiplets in the system
            all_chiplets: Set of all chiplet IDs used across all models
        """
        super().__init__(retired_mapped_models, global_time_us, num_chiplets, all_chiplets)
        self.approach_comparison_metrics = {
            "avg_time_per_input_per_model_type_main_sim": None,
            "avg_time_per_input_per_model_type_comparison": None
        }
    
    def compute_approach_comparison_metrics(self, individual_results: Optional[Dict] = None, 
                                           empty_system_individual_model_results: Optional[Dict] = None):
        """
        Computes model execution times and averages, including separate compute and communication metrics.
        
        Compares average execution time per input for each model type across three sources:
             1. Main simulation (self.retired_mapped_models)
             2. Individual model (main sim compute + communication from individual_results)
             3. Empty system model (from empty_system_individual_model_results)
        Also computes separate average compute and communication times per model type.
        Stores results in the `self.approach_comparison_metrics` attribute dictionary.
        
        Args:
            individual_results: Results from the individual layer communication simulation.
                               Expected format: {(model_idx, layer_idx): {"results": {'total_runtime_us': float}}}
                               If None, communication from this source is considered zero.
            empty_system_individual_model_results: Pre-calculated results for model types in an empty system.
                                                   Expected format: {model_name: {"latency": {
                                                       "total": float, "compute": float, "communication": float}}}
                                                   If None, metrics from this source will be None.
        Returns:
            dict: The `self.approach_comparison_metrics` dictionary containing "model_type_performance_comparison".
                  This key holds a breakdown for each model type, including:
                  - Main simulation metrics (avg total/compute/communication time per input, totals, instances)
                  - Individual model metrics (avg total time per input)
                  - Empty system model metrics (avg total/compute/communication time per input, total inputs)
        """
        
        # ====================================================
        # Initialize data structures for aggregating model metrics by type
        # ====================================================
        main_simulation_aggregated_data = {}  # Aggregates data from main simulation by model type
        individual_model_aggregated_data = {}  # Aggregates individual model data by model type

        # ====================================================
        # Process each model instance to extract timing data
        # ====================================================
        for model_idx, mapped_model in self.retired_mapped_models.items():
            # Initialize timing accumulators for this model instance
            model_instance_compute_time = 0.0
            model_instance_comm_time = 0.0

            # Get model type name for grouping
            model_type_name = mapped_model.model_name if hasattr(mapped_model, 'model_name') else f"Model {model_idx}"

            # Initialize aggregated data structures for new model types
            if model_type_name not in main_simulation_aggregated_data:
                main_simulation_aggregated_data[model_type_name] = {
                    "total_time": 0.0, 
                    "total_compute_time": 0.0, 
                    "total_comm_time": 0.0, 
                    "total_inputs": 0, 
                    "model_instances": []
                }
            
            if model_type_name not in individual_model_aggregated_data:
                individual_model_aggregated_data[model_type_name] = {
                    "total_simulated_time": 0.0, 
                    "total_inputs_processed": 0
                }

            # ====================================================
            # Extract compute latencies from main simulation
            # ====================================================
            processed_inputs = set() 
            if hasattr(mapped_model, 'layer_compute_latency') and mapped_model.layer_compute_latency:
                for input_idx, layer_compute_latencies in mapped_model.layer_compute_latency.items():
                    processed_inputs.add(input_idx)
                    for compute_latency in layer_compute_latencies.values():
                        if compute_latency > 0:
                            model_instance_compute_time += compute_latency
            
            # ====================================================
            # Extract activation communication latencies from main simulation
            # ====================================================
            if hasattr(mapped_model, 'layer_activation_comm_latency') and mapped_model.layer_activation_comm_latency:
                for input_idx, layer_activation_comm_latencies in mapped_model.layer_activation_comm_latency.items():
                    processed_inputs.add(input_idx)
                    for layer_idx, main_sim_activation_comm_latency in layer_activation_comm_latencies.items():
                        if main_sim_activation_comm_latency >= 0:  # Skip invalid latencies (-1)
                            model_instance_comm_time += main_sim_activation_comm_latency

            # ====================================================
            # Extract weight loading latencies from main simulation
            # ====================================================
            if hasattr(mapped_model, 'layer_weight_loading_latency') and mapped_model.layer_weight_loading_latency:
                for input_idx, layer_weight_loading_latencies in mapped_model.layer_weight_loading_latency.items():
                    processed_inputs.add(input_idx)
                    for layer_idx, main_sim_weight_loading_latency in layer_weight_loading_latencies.items():
                        if main_sim_weight_loading_latency >= 0:  # Skip invalid latencies (-1)
                            model_instance_comm_time += main_sim_weight_loading_latency
            
            # ====================================================
            # Determine number of inputs processed by this model instance
            # ====================================================
            num_inputs_processed = len(processed_inputs) if processed_inputs else \
                                  (mapped_model.num_inputs if hasattr(mapped_model, 'num_inputs') and mapped_model.num_inputs > 0 else 1)
            if num_inputs_processed == 0: 
                num_inputs_processed = 1

            # ====================================================
            # Aggregate main simulation data by model type
            # ====================================================
            model_instance_total_time = model_instance_compute_time + model_instance_comm_time
            
            main_simulation_aggregated_data[model_type_name]["total_time"] += model_instance_total_time
            main_simulation_aggregated_data[model_type_name]["total_compute_time"] += model_instance_compute_time
            main_simulation_aggregated_data[model_type_name]["total_comm_time"] += model_instance_comm_time
            main_simulation_aggregated_data[model_type_name]["total_inputs"] += num_inputs_processed
            main_simulation_aggregated_data[model_type_name]["model_instances"].append(model_idx)

            # ====================================================
            # Extract individual model communication latencies (if available)
            # ====================================================
            individual_model_comm_time = 0.0
            if hasattr(mapped_model, 'layer_compute_latency') and mapped_model.layer_compute_latency:
                # Iterate through computed layers to sum up their individual communication latencies
                for input_idx, computed_layers in mapped_model.layer_compute_latency.items():
                    for layer_idx, compute_latency_value in computed_layers.items():
                        if compute_latency_value > 0:  # If layer was actually computed for this input
                            individual_results_key = (model_idx, layer_idx) 
                            if individual_results:
                                individual_layer_data = individual_results.get(individual_results_key)
                                if individual_layer_data and "results" in individual_layer_data:
                                    individual_comm_latency = individual_layer_data["results"].get('total_runtime_us')
                                    if individual_comm_latency is not None and individual_comm_latency >= 0:
                                        individual_model_comm_time += individual_comm_latency
            
            # ====================================================
            # Aggregate individual model data by model type
            # ====================================================
            individual_model_total_time = model_instance_compute_time + individual_model_comm_time
            
            individual_model_aggregated_data[model_type_name]["total_simulated_time"] += individual_model_total_time
            individual_model_aggregated_data[model_type_name]["total_inputs_processed"] += num_inputs_processed

        # ====================================================
        # Calculate average metrics for main simulation data
        # ====================================================
        main_simulation_averages_by_type = {}
        for model_type_name, aggregated_data in main_simulation_aggregated_data.items():
            total_inputs = aggregated_data["total_inputs"]
            if total_inputs > 0:
                avg_total_time = aggregated_data["total_time"] / total_inputs
                avg_compute_time = aggregated_data["total_compute_time"] / total_inputs
                avg_comm_time = aggregated_data["total_comm_time"] / total_inputs
            else:
                avg_total_time = avg_compute_time = avg_comm_time = 0.0
                
            main_simulation_averages_by_type[model_type_name] = {
                "avg_time_per_input": avg_total_time, 
                "avg_compute_time_per_input": avg_compute_time, 
                "avg_comm_time_per_input": avg_comm_time, 
                "total_inputs": total_inputs, 
                "total_time": aggregated_data["total_time"], 
                "total_compute_time": aggregated_data["total_compute_time"],
                "total_comm_time": aggregated_data["total_comm_time"], 
                "model_instances": aggregated_data["model_instances"]
            }

        # ====================================================
        # Calculate average metrics for individual model data
        # ====================================================
        individual_model_averages_by_type = {}
        # Calculate averages from individual_model_aggregated_data.
        # This data includes model_instance_compute_time + individual_model_comm_time,
        # where individual_model_comm_time is 0 if individual_results was None.
        for model_type_name, aggregated_data in individual_model_aggregated_data.items():
            total_inputs_processed = aggregated_data["total_inputs_processed"]
            if total_inputs_processed > 0:
                avg_execution_time = aggregated_data["total_simulated_time"] / total_inputs_processed
                individual_model_averages_by_type[model_type_name] = avg_execution_time
            else:
                individual_model_averages_by_type[model_type_name] = None 
        
        # ====================================================
        # Create comprehensive performance comparison across all approaches
        # ====================================================
        comprehensive_approach_comparison = {}
        all_model_type_names = set(main_simulation_averages_by_type.keys()) | \
                                 set(individual_model_averages_by_type.keys())
        if empty_system_individual_model_results:
            all_model_type_names.update(empty_system_individual_model_results.keys())

        for model_type_name in all_model_type_names:
            # Get main simulation data (with detailed breakdown)
            main_sim_data = main_simulation_averages_by_type.get(model_type_name, {})
            
            # Extract individual model average time
            if individual_results is None:
                individual_model_avg_time = 0.0  # Set to 0.0 if no individual results were provided
            else:
                # Otherwise, use the calculated average from individual_model_averages_by_type
                # This might be None if total_inputs_processed was 0 for this model_type_name.
                individual_model_avg_time = individual_model_averages_by_type.get(model_type_name)
            
            # Extract empty system model metrics
            empty_system_model_metrics = {
                "avg_time_per_input": None,
                "avg_compute_time_per_input": None,
                "avg_comm_time_per_input": None,
                "total_inputs": None  # This will remain None as it's not in the source structure
            }
            if empty_system_individual_model_results:
                # Get data for the specific model type from the empty system results
                model_data_for_empty_system = empty_system_individual_model_results.get(model_type_name, {})
                
                # Get the 'latency' sub-dictionary, defaulting to an empty dict if not found
                latency_values_for_empty_system = model_data_for_empty_system.get("latency", {})
                
                # Ensure latency_values_for_empty_system is a dictionary before trying to get items from it
                if isinstance(latency_values_for_empty_system, dict):
                    empty_system_model_metrics["avg_time_per_input"] = latency_values_for_empty_system.get("total")
                    empty_system_model_metrics["avg_compute_time_per_input"] = latency_values_for_empty_system.get("compute")
                    empty_system_model_metrics["avg_comm_time_per_input"] = latency_values_for_empty_system.get("communication")
                
                # 'total_inputs' is not available in the source data structure for empty system results,
                # so empty_system_model_metrics["total_inputs"] remains None as initialized.

            # Store comprehensive comparison data with detailed main simulation breakdown
            comprehensive_approach_comparison[model_type_name] = {
                # Main simulation detailed metrics (includes compute/communication breakdown)
                "main_simulation": {
                    "avg_time_per_input": main_sim_data.get("avg_time_per_input"),
                    "avg_compute_time_per_input": main_sim_data.get("avg_compute_time_per_input"),
                    "avg_comm_time_per_input": main_sim_data.get("avg_comm_time_per_input"),
                    "total_inputs": main_sim_data.get("total_inputs"),
                    "total_time": main_sim_data.get("total_time"),
                    "total_compute_time": main_sim_data.get("total_compute_time"),
                    "total_comm_time": main_sim_data.get("total_comm_time"),
                    "model_instances": main_sim_data.get("model_instances")
                },
                # Individual model metrics
                "individual_model": {
                    "avg_time_per_input": individual_model_avg_time
                },
                # Empty system model metrics
                "empty_system_model": empty_system_model_metrics
            }
            
        # ====================================================
        # Store results in instance attributes (single comprehensive metric)
        # ====================================================
        self.approach_comparison_metrics["model_type_performance_comparison"] = comprehensive_approach_comparison

        return self.approach_comparison_metrics
