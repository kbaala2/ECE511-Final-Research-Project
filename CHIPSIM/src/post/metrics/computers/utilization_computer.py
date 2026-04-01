"""
Utilization Computer Module

This module contains the UtilizationComputer class responsible for computing
chiplet utilization metrics using two different methods:

1. Method 1: Latency-based aggregation
   - Sums all latencies per chiplet
   - Computes utilization as percentage of total simulation time
   - Does not account for overlapping execution

2. Method 2: Time-based activity tracking
   - Tracks chiplet activity at discrete time steps
   - Accounts for overlapping execution
   - Provides detailed time-series utilization data
"""

import numpy as np
from typing import Any, Dict, List, Set
from .base_computer import BaseMetricComputer


class UtilizationComputer(BaseMetricComputer):
    """
    Computes utilization metrics for chiplets using two different methods.
    
    Method 1 (compute_avg_system_utilization): 
        Aggregate latency-based utilization
    Method 2 (compute_utilization_over_time): 
        Time-based activity tracking with discrete sampling
    """

    def compute_avg_system_utilization(self):
        """
        Computes the system compute utilization per chiplet based on total compute time.
        
        This method calculates utilization as the percentage of time each chiplet
        spends on compute, communication, or combined (compute + communication) tasks
        over the total simulation time. It achieves this by summing all relevant latencies
        (compute, communication, or their sum for combined) for each chiplet across all
        models and dividing by the total simulation time.

        It updates the following instance attributes:
        - For Compute Time:
            - `self.chiplet_total_compute_time_us`
            - `self.per_chiplet_compute_time_utilization_pct`
            - `self.mean_compute_time_chiplet_utilization_pct`
        - For Communication Time:
            - `self.chiplet_total_communication_time_us`
            - `self.per_chiplet_communication_time_utilization_pct`
            - `self.mean_communication_time_chiplet_utilization_pct`
        - For Combined Time (sum of compute and communication task durations):
            - `self.chiplet_total_combined_busy_time_us`
            - `self.per_chiplet_combined_time_utilization_pct`
            - `self.mean_combined_time_chiplet_utilization_pct`

        Note: This method sums latencies and does not account for overlapping execution
        intervals in the same way the `compute_utilization_over_time` method does.
        The assertion regarding 'layer_start_time_us' and 'layer_completion_time_us'
        is primarily for ensuring that models have detailed timing info if expected,
        but these specific start/completion times are not directly used in the latency summation here.

        Returns:
            None. Results are stored in instance attributes.
        """

        # Initialize dictionaries to store total times for each chiplet
        chiplet_total_compute_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_activation_comm_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_weight_loading_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_combined_busy_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}

        # Iterate through each model
        for model_idx, mapped_model in self.retired_mapped_models.items():
            # Check if the model has per-input, per-layer start/completion times
            # (these are available when parallelization is enabled)
            # Also check for latency information.
            required_attrs = ['layer_start_time_us', 'layer_completion_time_us', 
                              'layer_compute_latency', 'layer_activation_comm_latency', 'layer_weight_loading_latency']
            for attr in required_attrs:
                assert hasattr(mapped_model, attr), \
                       f"Model {model_idx} ({mapped_model.model_name if hasattr(mapped_model, 'model_name') else 'Unknown'}) is missing '{attr}' attribute. This is required for utilization calculation."

            # Get mapping details
            layer_to_chiplets = {}
            for layer_idx, mappings in mapped_model.mapping:
                layer_to_chiplets[layer_idx] = mappings

            # Use detailed layer timing for more accurate utilization calculation
            for input_idx in range(mapped_model.num_inputs):
                for layer_idx in range(mapped_model.num_layers):
                    compute_latency = mapped_model.layer_compute_latency[input_idx].get(layer_idx, 0.0)
                    activation_comm_latency = mapped_model.layer_activation_comm_latency[input_idx].get(layer_idx, 0.0)
                    weight_loading_latency = mapped_model.layer_weight_loading_latency[input_idx].get(layer_idx, 0.0)

                    # Validate latencies
                    if not isinstance(compute_latency, (int, float)): compute_latency = 0.0
                    if not isinstance(activation_comm_latency, (int, float)): activation_comm_latency = 0.0
                    if not isinstance(weight_loading_latency, (int, float)): weight_loading_latency = 0.0
                    compute_latency = max(0.0, compute_latency)
                    activation_comm_latency = max(0.0, activation_comm_latency)
                    weight_loading_latency = max(0.0, weight_loading_latency)

                    # Process if there's any activity for this layer
                    if not (compute_latency > 0 or activation_comm_latency > 0 or weight_loading_latency > 0):
                        continue # Skip if all are zero

                    # Check if this layer has chiplet mappings
                    if layer_idx in layer_to_chiplets:
                        # Add time contribution for each chiplet involved
                        for chiplet_id, percentage in layer_to_chiplets[layer_idx]:
                            if chiplet_id not in self.all_chiplets: # Should not happen if all_chiplets is comprehensive
                                print(f"Warning: Chiplet ID {chiplet_id} from mapping not in self.all_chiplets. Skipping.")
                                continue
                            
                            if compute_latency > 0:
                                chiplet_total_compute_time_us[chiplet_id] += compute_latency
                            if activation_comm_latency > 0:
                                chiplet_total_activation_comm_time_us[chiplet_id] += activation_comm_latency
                            if weight_loading_latency > 0:
                                chiplet_total_weight_loading_time_us[chiplet_id] += weight_loading_latency
                            # Combined busy time adds all three, as they are sequential components of work for this task segment
                            chiplet_total_combined_busy_time_us[chiplet_id] += (compute_latency + activation_comm_latency + weight_loading_latency)

        # --- Store total times ---
        self.chiplet_total_compute_time_us = chiplet_total_compute_time_us
        self.chiplet_total_activation_comm_time_us = chiplet_total_activation_comm_time_us
        self.chiplet_total_weight_loading_time_us = chiplet_total_weight_loading_time_us
        self.chiplet_total_combined_busy_time_us = chiplet_total_combined_busy_time_us

        # --- Calculate and store per-chiplet utilization percentages ---
        per_chiplet_compute_utilization_pct = {}
        for chiplet_id, compute_time in chiplet_total_compute_time_us.items():
            util = (compute_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_compute_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_compute_time_utilization_pct = per_chiplet_compute_utilization_pct

        per_chiplet_activation_comm_utilization_pct = {}
        for chiplet_id, activation_comm_time in chiplet_total_activation_comm_time_us.items():
            util = (activation_comm_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_activation_comm_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_activation_comm_time_utilization_pct = per_chiplet_activation_comm_utilization_pct

        per_chiplet_weight_loading_utilization_pct = {}
        for chiplet_id, weight_loading_time in chiplet_total_weight_loading_time_us.items():
            util = (weight_loading_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_weight_loading_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_weight_loading_time_utilization_pct = per_chiplet_weight_loading_utilization_pct

        per_chiplet_combined_utilization_pct = {}
        for chiplet_id, combined_time in chiplet_total_combined_busy_time_us.items():
            util = (combined_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_combined_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_combined_time_utilization_pct = per_chiplet_combined_utilization_pct

        # --- Assertions for accumulated times ---
        if self.retired_mapped_models:
            total_accumulated_compute = sum(chiplet_total_compute_time_us.values())
            total_accumulated_activation_comm = sum(chiplet_total_activation_comm_time_us.values())
            total_accumulated_weight_loading = sum(chiplet_total_weight_loading_time_us.values())
            total_accumulated_combined = sum(chiplet_total_combined_busy_time_us.values())

            # Check if any activity was expected to be recorded
            # This assertion might be too strict if a sim genuinely has zero activity for a type.
            # For now, let's ensure that if there *was* activity, util isn't unexpectedly zero.
            if any(val > 0 for val in chiplet_total_compute_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_compute_utilization_pct.values()), \
                     "Non-zero total compute time but all per-chiplet compute utilizations are zero."
            if any(val > 0 for val in chiplet_total_activation_comm_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_activation_comm_utilization_pct.values()), \
                     "Non-zero total activation communication time but all per-chiplet activation communication utilizations are zero."
            if any(val > 0 for val in chiplet_total_weight_loading_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_weight_loading_utilization_pct.values()), \
                     "Non-zero total weight loading time but all per-chiplet weight loading utilizations are zero."
            if any(val > 0 for val in chiplet_total_combined_busy_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_combined_utilization_pct.values()), \
                     "Non-zero total combined time but all per-chiplet combined utilizations are zero."

        # --- Calculate and store mean of per-chiplet utilizations ---
        if per_chiplet_compute_utilization_pct:
            self.mean_compute_time_chiplet_utilization_pct = sum(per_chiplet_compute_utilization_pct.values()) / len(per_chiplet_compute_utilization_pct)
        else:
            self.mean_compute_time_chiplet_utilization_pct = 0.0 # Or -1.0, consistent with previous behavior

        if per_chiplet_activation_comm_utilization_pct:
            self.mean_activation_comm_time_chiplet_utilization_pct = sum(per_chiplet_activation_comm_utilization_pct.values()) / len(per_chiplet_activation_comm_utilization_pct)
        else:
            self.mean_activation_comm_time_chiplet_utilization_pct = 0.0

        if per_chiplet_weight_loading_utilization_pct:
            self.mean_weight_loading_time_chiplet_utilization_pct = sum(per_chiplet_weight_loading_utilization_pct.values()) / len(per_chiplet_weight_loading_utilization_pct)
        else:
            self.mean_weight_loading_time_chiplet_utilization_pct = 0.0

        if per_chiplet_combined_utilization_pct:
            self.mean_combined_time_chiplet_utilization_pct = sum(per_chiplet_combined_utilization_pct.values()) / len(per_chiplet_combined_utilization_pct)
        else:
            self.mean_combined_time_chiplet_utilization_pct = 0.0

        # Add combined communication utilization for cross-simulation compatibility
        # This combines activation communication and weight loading utilization
        if per_chiplet_activation_comm_utilization_pct and per_chiplet_weight_loading_utilization_pct:
            combined_comm_utilization = {}
            for chiplet_id in per_chiplet_activation_comm_utilization_pct:
                activation_comm_util = per_chiplet_activation_comm_utilization_pct.get(chiplet_id, 0.0)
                weight_loading_util = per_chiplet_weight_loading_utilization_pct.get(chiplet_id, 0.0)
                combined_comm_utilization[chiplet_id] = activation_comm_util + weight_loading_util
            self.mean_communication_time_chiplet_utilization_pct = sum(combined_comm_utilization.values()) / len(combined_comm_utilization)
        else:
            self.mean_communication_time_chiplet_utilization_pct = 0.0

        return # Method updates instance attributes, returns None

    def compute_utilization_over_time(self, time_step_us=1.0):
        """
        Computes chiplet COMPUTE, COMMUNICATION, and COMBINED activity at each discrete time step
        and derives utilization metrics, storing the results in instance attributes.

        This method calculates:
        1.  Activity of each chiplet at each time step for compute, communication, and combined states.
        2.  Aggregated activity over time for each chiplet.
        3.  System-wide utilization percentage at each time point.
        4.  Average utilization percentage per chiplet over the simulation duration.
        5.  Overall average system utilization percentage across all time steps.

        Updates the following instance attributes:
            - `self.utilization_time_points`: Shared time points for all utilization calculations.
            - Compute Metrics:
                - `self.chiplet_compute_activity_at_time_step`
                - `self.chiplet_compute_activity_over_time`
                - `self.system_compute_utilization_over_time_pct`
            - Communication Metrics:
                - `self.chiplet_communication_activity_at_time_step`
                - `self.chiplet_communication_activity_over_time`
                - `self.system_communication_utilization_over_time_pct`
            - Combined Metrics:
                - `self.chiplet_combined_activity_at_time_step`
                - `self.chiplet_combined_activity_over_time`
                - `self.system_combined_utilization_over_time_pct`

        Args:
            time_step_us (float): Time step in microseconds for sampling utilization.

        Returns:
            None
        """
        
        # --- Helper to set all utilization attributes to default/empty ---
        def _set_default_utilization_metrics(time_points_val=np.array([]), activity_val={}, list_val=[], dict_val={}, float_val=0.0, num_tp=0, chiplet_ids_for_avg=None):
            if chiplet_ids_for_avg is None:
                chiplet_ids_for_avg = set()

            self.utilization_time_points = time_points_val
            
            self.chiplet_compute_activity_at_time_step = activity_val
            self.chiplet_compute_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_compute_utilization_over_time_pct = list_val
            
            self.chiplet_activation_comm_activity_at_time_step = activity_val
            self.chiplet_activation_comm_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_activation_comm_utilization_over_time_pct = list_val
            
            self.chiplet_weight_loading_activity_at_time_step = activity_val
            self.chiplet_weight_loading_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_weight_loading_utilization_over_time_pct = list_val
            
            self.chiplet_combined_activity_at_time_step = activity_val
            self.chiplet_combined_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_combined_utilization_over_time_pct = list_val

        # --- Handle Empty Input or Invalid Parameters ---
        if not self.retired_mapped_models or self.global_time_us <= 0 or time_step_us <= 0:
            _set_default_utilization_metrics()
            return

        # --- Build the set of all unique chiplet IDs involved ---
        all_system_chiplet_ids = set()
        total_system_chiplets_config = self.num_chiplets
        if total_system_chiplets_config <= 0:
            print(f"Warning: Total configured chiplets reported as {total_system_chiplets_config}. Check MetricComputer initialization. This may affect system utilization percentages.")
            total_system_chiplets_config = 0  # Treat as invalid for calculation if necessary

        for model in self.retired_mapped_models.values():
            if not hasattr(model, 'mapping'):
                print(f"Warning: Model {model.model_idx if hasattr(model, 'model_idx') else 'Unknown'} is missing 'mapping' attribute. Skipping for chiplet ID collection.")
                continue
            for layer_idx, mappings in model.mapping:
                for chiplet_id, percentage in mappings:
                    all_system_chiplet_ids.add(chiplet_id)

        # --- Determine effective total chiplets for utilization denominator ---
        if total_system_chiplets_config > 0:
            effective_total_chiplets_for_util = total_system_chiplets_config
        else:
            print("Warning: Using count of *used* chiplets for system utilization denominator due to invalid or zero configured total.")
            effective_total_chiplets_for_util = len(all_system_chiplet_ids)


        # --- Handle case where no chiplets are found or effective total is zero ---
        if not all_system_chiplet_ids:
            print("Warning: No chiplets found in any model mappings. All utilization metrics will be zero or empty.")
            # Time points are still generated, system utilization is 0%, per-chiplet metrics are empty.
            time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
            num_time_points = len(time_points)
            _set_default_utilization_metrics(
                time_points_val=time_points,
                activity_val={tp: {} for tp in time_points}, # Activity is empty per time point
                list_val=[0.0] * num_time_points if num_time_points > 0 else [],
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # No chiplets, so empty
                float_val=0.0,
                num_tp=num_time_points,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            return

        # --- Create time points ---
        time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
        self.utilization_time_points = time_points
        num_time_points = len(time_points)

        if num_time_points == 0:
            print("Warning: No time points generated (global_time_us might be too small or time_step_us invalid). All utilization metrics will be zero or empty.")
            _set_default_utilization_metrics(
                time_points_val=time_points, # empty np.array
                activity_val={},
                list_val=[],
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # Avg util is 0 if no time
                float_val=0.0,
                num_tp=0,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            return

        if effective_total_chiplets_for_util == 0:
            print("Warning: effective_total_chiplets_for_util is zero. System utilization cannot be computed accurately and will be reported as 0%.")
            # Chiplet activity can still be tracked, but system-wide utilization is ill-defined or zero.
            _set_default_utilization_metrics(
                time_points_val=time_points,
                activity_val={tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points},
                list_val=[0.0] * num_time_points, # System util is 0
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # Chiplet avg can still be calculated based on its own activity
                float_val=0.0, # System avg is 0
                num_tp=num_time_points,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            # Continue to calculate per-chiplet activity, but system metrics will be 0.
            # The _calculate_derived_utilization_metrics helper will handle division by zero for system util.

        # --- Initialize primary activity dictionaries ---
        chiplet_compute_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_activation_comm_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_weight_loading_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_combined_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        
        epsilon = 1e-9 # For ceiling operations

        # --- Populate activity dictionaries by iterating through computations and communications ---
        for model_idx, mapped_model in self.retired_mapped_models.items():
            if not (hasattr(mapped_model, 'mapping') and
                    hasattr(mapped_model, 'num_inputs') and
                    hasattr(mapped_model, 'num_layers') and
                    hasattr(mapped_model, 'layer_start_time_us') and
                    hasattr(mapped_model, 'layer_compute_latency') and # Needed for compute and combined
                    hasattr(mapped_model, 'layer_activation_comm_latency') and # Needed for activation communication and combined
                    hasattr(mapped_model, 'layer_weight_loading_latency')): # Needed for weight loading and combined
                print(f"Warning: Model {model_idx} ({mapped_model.model_name if hasattr(mapped_model, 'model_name') else 'Unknown'}) is missing required attributes for detailed utilization calculation. Skipping this model.")
                continue

            layer_to_chiplets_map = {layer_idx: mappings for layer_idx, mappings in mapped_model.mapping}

            for input_idx in range(mapped_model.num_inputs):
                # Check if timing dictionaries exist for this input_idx
                if not (input_idx in mapped_model.layer_start_time_us and
                        input_idx in mapped_model.layer_compute_latency and
                        input_idx in mapped_model.layer_activation_comm_latency and
                        input_idx in mapped_model.layer_weight_loading_latency):
                    # This might happen if an input didn't run fully or data is missing
                    continue

                for layer_idx in range(mapped_model.num_layers):
                    layer_start_time = mapped_model.layer_start_time_us[input_idx].get(layer_idx, -1)
                    compute_latency = mapped_model.layer_compute_latency[input_idx].get(layer_idx, 0)
                    activation_comm_latency = mapped_model.layer_activation_comm_latency[input_idx].get(layer_idx, 0)
                    weight_loading_latency = mapped_model.layer_weight_loading_latency[input_idx].get(layer_idx, 0)

                    # Basic validation of timing data
                    if not (isinstance(layer_start_time, (int, float)) and
                            isinstance(compute_latency, (int, float)) and
                            isinstance(activation_comm_latency, (int, float)) and
                            isinstance(weight_loading_latency, (int, float))):
                        # print(f"Warning: Invalid timing data types for Model {model_idx}, Input {input_idx}, Layer {layer_idx}. Skipping.")
                        continue
                    
                    if compute_latency < 0: compute_latency = 0 # Treat negative as zero
                    if activation_comm_latency < 0: activation_comm_latency = 0 # Treat negative as zero
                    if weight_loading_latency < 0: weight_loading_latency = 0 # Treat negative as zero

                    chiplets_for_layer = {cid for cid, _ in layer_to_chiplets_map.get(layer_idx, [])}
                    if not chiplets_for_layer:
                        continue

                    # --- Compute Activity ---
                    if layer_start_time >= 0 and compute_latency > 0:
                        layer_compute_end_time = layer_start_time + compute_latency
                        start_tp_idx = max(0, int(np.floor(layer_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((layer_compute_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_compute_activity_at_time_step[tp]:
                                    chiplet_compute_activity_at_time_step[tp][cid] = 1
                    
                    # --- Activation Communication Activity ---
                    if layer_start_time >= 0 and activation_comm_latency > 0: # compute_latency can be 0 if only communication happens after a data arrival
                        activation_comm_start_time = layer_start_time + compute_latency # Activation comm starts after compute finishes
                        activation_comm_end_time = activation_comm_start_time + activation_comm_latency
                        start_tp_idx = max(0, int(np.floor(activation_comm_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((activation_comm_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_activation_comm_activity_at_time_step[tp]:
                                    chiplet_activation_comm_activity_at_time_step[tp][cid] = 1

                    # --- Weight Loading Communication Activity ---
                    if layer_start_time >= 0 and weight_loading_latency > 0:
                        # Weight loading can occur at any time, not necessarily after compute
                        # For now, assume it occurs before compute (typical weight loading pattern)
                        weight_loading_start_time = layer_start_time - weight_loading_latency
                        weight_loading_end_time = layer_start_time
                        start_tp_idx = max(0, int(np.floor(weight_loading_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((weight_loading_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_weight_loading_activity_at_time_step[tp]:
                                    chiplet_weight_loading_activity_at_time_step[tp][cid] = 1

                    # --- Combined Activity (Compute OR Activation Communication OR Weight Loading) ---
                    # A chiplet is active if it's computing OR doing activation communication OR weight loading.
                    # The interval spans from weight loading start to end of activation communication.
                    if layer_start_time >= 0 and (compute_latency > 0 or activation_comm_latency > 0 or weight_loading_latency > 0):
                        # Combined activity spans from weight loading start to activation communication end
                        activity_start_time = layer_start_time - weight_loading_latency  # Start with weight loading
                        activity_end_time = layer_start_time + compute_latency + activation_comm_latency  # End with activation communication
                        
                        start_tp_idx = max(0, int(np.floor(activity_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((activity_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_combined_activity_at_time_step[tp]:
                                    chiplet_combined_activity_at_time_step[tp][cid] = 1
        
        # --- Calculate Derived Metrics ---
        (self.chiplet_compute_activity_over_time,
         self.system_compute_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_compute_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_compute_activity_at_time_step = chiplet_compute_activity_at_time_step

        (self.chiplet_activation_comm_activity_over_time,
         self.system_activation_comm_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_activation_comm_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_activation_comm_activity_at_time_step = chiplet_activation_comm_activity_at_time_step

        (self.chiplet_weight_loading_activity_over_time,
         self.system_weight_loading_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_weight_loading_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_weight_loading_activity_at_time_step = chiplet_weight_loading_activity_at_time_step

        (self.chiplet_combined_activity_over_time,
         self.system_combined_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_combined_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_combined_activity_at_time_step = chiplet_combined_activity_at_time_step
        
        return # Explicitly return None

    def _calculate_derived_utilization_metrics(self, activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points):
        """
        Helper to calculate derived utilization metrics from activity data.
        
        Args:
            activity_at_time_step: Dictionary mapping time points to chiplet activity
            time_points: Array of time points
            all_system_chiplet_ids: Set of all chiplet IDs in the system
            effective_total_chiplets_for_util: Total number of chiplets for utilization denominator
            num_time_points: Number of time points
            
        Returns:
            Tuple of (activity_over_time, system_util_over_time_pct)
        """
        if num_time_points == 0: # Should be caught earlier, but as a safeguard
            activity_over_time = {cid: [] for cid in all_system_chiplet_ids}
            system_util_over_time_pct = []
            return activity_over_time, system_util_over_time_pct

        activity_over_time = {cid: [0] * num_time_points for cid in all_system_chiplet_ids}
        system_util_over_time_pct = [0.0] * num_time_points

        for t_idx, tp in enumerate(time_points):
            active_chiplets_count_at_tp = 0
            activity_dict_at_tp = activity_at_time_step.get(tp, {}) # Safe get
            
            for cid in all_system_chiplet_ids:
                activity = activity_dict_at_tp.get(cid, 0) # Safe get
                activity_over_time[cid][t_idx] = activity
                active_chiplets_count_at_tp += activity
            
            if effective_total_chiplets_for_util > 0:
                system_util_over_time_pct[t_idx] = (active_chiplets_count_at_tp / effective_total_chiplets_for_util) * 100.0
            else: # Denominator is zero, system utilization is undefined or 0%
                system_util_over_time_pct[t_idx] = 0.0

        return activity_over_time, system_util_over_time_pct
