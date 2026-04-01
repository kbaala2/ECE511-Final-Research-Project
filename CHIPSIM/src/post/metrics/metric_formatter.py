import os
from typing import Any, Dict, List, Optional

# Import MetricNode from new module (Phase 1 refactoring)
from .metric_node import MetricNode

# --- MetricFormatter Class ---

class MetricFormatter:
    def __init__(self, metric_computer: Any, global_manager: Any):
        self.mc = metric_computer
        self.gm = global_manager

    def format_all_model_metrics(self) -> MetricNode:
        """
        Runs the per-model metric formatting for each retired model in the global manager.
        Returns:
            MetricNode: A root node containing all model metrics as children
        """
        # Use post_warmup_retired_models if available, otherwise use retired_mapped_models
        retired_models = getattr(self.gm, 'post_warmup_retired_models', None)
        if not retired_models:
            retired_models = self.gm.retired_mapped_models
        
        children = []
        for model_idx, mapped_model in retired_models.items():
            child_node = self._format_per_model_metrics(model_idx, mapped_model)
            children.append(child_node)
        return MetricNode(
            section="All Models", 
            name="All Models", 
            children=children,
            save_children_separately=True  # Save each model as a separate file
        )

    def _format_per_model_metrics(self, model_idx: int, mapped_model: Any) -> MetricNode:
        """
        Returns metrics for a single model using MetricNode.
        Args:
            model_idx: Index of the model to format metrics for
            mapped_model: The MappedModel object
        Returns:
            MetricNode: Formatted metrics for the model
        """
        retired_models = getattr(self.gm, 'post_warmup_retired_models', self.gm.retired_mapped_models)
        children = []
        # Calculate model duration
        model_duration = -1.0
        if mapped_model.model_completion_time_us > 0:
            model_duration = mapped_model.model_completion_time_us - mapped_model.model_start_time_us
            
        # ================================================================
        # Section: Model Overall Times (moved to first)
        # ================================================================
        section = "Model Overall Times"
        model_start = getattr(mapped_model, 'model_start_time_us', -1)
        model_end = getattr(mapped_model, 'model_completion_time_us', -1)
        children.append(MetricNode(
            section=section,
            name="Overall Times",
            values={
                "Start": model_start,
                "End": model_end,
                "Duration": model_duration
            },
            unit="μs"
        ))

        # ================================================================
        # Section: Input Start/End Times (as table)
        # ================================================================
        section = "Input Start/End Times"
        num_inputs = getattr(mapped_model, 'num_inputs', 0)
        start_times_list = getattr(mapped_model, 'input_start_time_us', [])
        completion_times_list = getattr(mapped_model, 'input_completion_time_us', [])
        input_times_rows = []
        for input_idx in range(num_inputs):
            start_time = start_times_list[input_idx] if input_idx < len(start_times_list) else -1
            end_time = completion_times_list[input_idx] if input_idx < len(completion_times_list) else -1
            input_times_rows.append({
                "Input": int(input_idx),
                "Start (μs)": start_time,
                "End (μs)": end_time
            })
        
        if input_times_rows:
            children.append(MetricNode(
                section=section,
                name="Input Times Table",
                columns=["Input", "Start (μs)", "End (μs)"],
                rows=input_times_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Input Times Table",
                description="No input timing data available."
            ))
            
        # ================================================================
        # Section: Summed Time Per Input (as table)
        # ================================================================
        section = "Summed Time Per Input"
        compute_latency_dict = getattr(mapped_model, 'layer_compute_latency', {})
        activation_comm_latency_dict = getattr(mapped_model, 'layer_activation_comm_latency', {})
        
        summed_time_rows = []
        for input_idx in range(num_inputs):
            input_compute_time = sum(latency for latency in compute_latency_dict.get(input_idx, {}).values() if latency > 0)
            input_activation_comm_time = sum(latency for latency in activation_comm_latency_dict.get(input_idx, {}).values() if latency > 0)
            
            # Calculate weight loading time for this specific input
            input_weight_loading_time = 0.0
            if hasattr(mapped_model, 'phases') and hasattr(mapped_model, 'phase_instances'):
                if input_idx in mapped_model.phase_instances:
                    for phase_id, phase in mapped_model.phases.items():
                        if hasattr(phase, 'layers_to_load'):  # WeightLoadingPhase
                            instance = mapped_model.phase_instances[input_idx].get(phase_id)
                            if instance and instance.latency_us > 0:
                                input_weight_loading_time += instance.latency_us
            
            input_total_time = input_compute_time + input_activation_comm_time + input_weight_loading_time
            summed_time_rows.append({
                "Input": int(input_idx),
                "Compute (μs)": input_compute_time,
                "Activation Communication (μs)": input_activation_comm_time,
                "Weight Loading (μs)": input_weight_loading_time,
                "Total (μs)": input_total_time
            })
        
        if summed_time_rows:
            children.append(MetricNode(
                section=section,
                name="Summed Time Table",
                columns=["Input", "Compute (μs)", "Activation Communication (μs)", "Weight Loading (μs)", "Total (μs)"],
                rows=summed_time_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Summed Time Table",
                description="No summed time data available."
            ))
        
        # ================================================================
        # Section: Layer Latency per input
        # ================================================================
        section = "Layer Latency per input"
        activation_comm_latency_dict = getattr(mapped_model, 'layer_activation_comm_latency', {})
        latency_table_rows = []
        for input_idx in range(num_inputs):
            layers_for_input = set()
            if input_idx in compute_latency_dict: layers_for_input.update(compute_latency_dict[input_idx].keys())
            if input_idx in activation_comm_latency_dict: layers_for_input.update(activation_comm_latency_dict[input_idx].keys())
            for layer_idx in sorted(list(layers_for_input)):
                row = {
                    "Input": int(input_idx),
                    "Layer": int(layer_idx),
                    "Compute Latency (μs)": compute_latency_dict.get(input_idx, {}).get(layer_idx, -1.0),
                    "Activation Comm Latency (μs)": activation_comm_latency_dict.get(input_idx, {}).get(layer_idx, -1.0),
                }
                latency_table_rows.append(row)
        if latency_table_rows:
            children.append(MetricNode(
                section=section,
                name="Latency Table",
                columns=["Input", "Layer", "Compute Latency (μs)", "Activation Comm Latency (μs)"],
                rows=latency_table_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Latency Table",
                description="No layer latency data found for any input."
            ))
            
        # ================================================================
        # Section: Phase Execution Schedule
        # ================================================================
        section = "Phase Execution Schedule"
        if hasattr(mapped_model, 'phases') and hasattr(mapped_model, 'phase_instances') and hasattr(mapped_model, 'phase_execution_order'):
            phase_schedule_rows = []
            for input_idx in range(num_inputs):
                if input_idx in mapped_model.phase_instances:
                    for phase_id in mapped_model.phase_execution_order:
                        phase = mapped_model.phases.get(phase_id)
                        instance = mapped_model.phase_instances[input_idx].get(phase_id)
                        if phase and instance:
                            # Determine phase type and description
                            if hasattr(phase, 'layer_idx'):  # ComputePhase or ActivationCommPhase
                                if hasattr(phase, 'layers_to_load'):  # This shouldn't happen, but just in case
                                    phase_type = "Weight Loading"
                                    phase_desc = f"Layers {phase.layers_to_load}"
                                else:
                                    phase_type = "Compute" if hasattr(phase, 'chiplet_assignments') else "Activation Communication"
                                    phase_desc = f"Layer {phase.layer_idx}"
                            elif hasattr(phase, 'layers_to_load'):  # WeightLoadingPhase
                                phase_type = "Weight Loading"
                                phase_desc = f"Layers {phase.layers_to_load}"
                            else:
                                phase_type = "Unknown"
                                phase_desc = f"Phase {phase_id}"
                            
                            phase_schedule_rows.append({
                                "Input": int(input_idx),
                                "Phase ID": int(phase_id),
                                "Type": phase_type,
                                "Description": phase_desc,
                                "Start (μs)": instance.start_time_us if instance.start_time_us >= 0 else -1,
                                "Duration (μs)": instance.latency_us if instance.latency_us > 0 else 0,
                                "Completion (μs)": instance.completion_time_us if instance.completion_time_us >= 0 else -1
                            })
            
            if phase_schedule_rows:
                children.append(MetricNode(
                    section=section,
                    name="Phase Schedule Table",
                    columns=["Input", "Phase ID", "Type", "Description", "Start (μs)", "Duration (μs)", "Completion (μs)"],
                    rows=phase_schedule_rows
                ))
            else:
                children.append(MetricNode(
                    section=section,
                    name="Phase Schedule Table",
                    description="No phase schedule data available."
                ))
        else:
            children.append(MetricNode(
                section=section,
                name="Phase Schedule Table",
                description="Phase schedule data not available."
            ))

        
        # ================================================================
        # Section: Weight Loading Schedule
        # ================================================================
        section = "Weight Loading Schedule"
        if hasattr(mapped_model, 'phases') and hasattr(mapped_model, 'phase_instances'):
            weight_loading_rows = []
            for input_idx in range(num_inputs):
                if input_idx in mapped_model.phase_instances:
                    for phase_id, phase in mapped_model.phases.items():
                        if hasattr(phase, 'layers_to_load'):  # WeightLoadingPhase
                            instance = mapped_model.phase_instances[input_idx].get(phase_id)
                            if instance and instance.latency_us > 0:
                                # Find next compute layer after this weight loading phase
                                next_compute_layer = "N/A"
                                if hasattr(mapped_model, 'phase_execution_order'):
                                    current_idx = mapped_model.phase_execution_order.index(phase_id)
                                    for next_phase_id in mapped_model.phase_execution_order[current_idx + 1:]:
                                        next_phase = mapped_model.phases.get(next_phase_id)
                                        if next_phase and hasattr(next_phase, 'layer_idx') and hasattr(next_phase, 'chiplet_assignments'):
                                            next_compute_layer = next_phase.layer_idx
                                            break
                                
                                weight_loading_rows.append({
                                    "Input": int(input_idx),
                                    "Op ID": int(phase_id),
                                    "Start (μs)": instance.start_time_us if instance.start_time_us >= 0 else -1,
                                    "Duration (μs)": instance.latency_us,
                                    "Completion (μs)": instance.completion_time_us if instance.completion_time_us >= 0 else -1,
                                    "Next Compute Layer": f"Input {input_idx}, Layer {next_compute_layer}" if next_compute_layer != "N/A" else "N/A"
                                })
            
            if weight_loading_rows:
                children.append(MetricNode(
                    section=section,
                    name="Weight Loading Schedule Table",
                    columns=["Input", "Op ID", "Start (μs)", "Duration (μs)", "Completion (μs)", "Next Compute Layer"],
                    rows=weight_loading_rows
                ))
            else:
                children.append(MetricNode(
                    section=section,
                    name="Weight Loading Schedule Table",
                    description="No weight loading operations found."
                ))
        else:
            children.append(MetricNode(
                section=section,
                name="Weight Loading Schedule Table",
                description="Weight loading schedule data not available."
            ))

        # ================================================================
        # Section: Weight Loading Traffic (bytes)
        # ================================================================
        section = "Weight Loading Traffic (bytes)"
        if hasattr(mapped_model, 'phases') and hasattr(mapped_model, 'phase_instances'):
            weight_loading_traffic_rows = []
            for input_idx in range(num_inputs):
                if input_idx in mapped_model.phase_instances:
                    for phase_id, phase in mapped_model.phases.items():
                        if hasattr(phase, 'layers_to_load'):  # WeightLoadingPhase
                            instance = mapped_model.phase_instances[input_idx].get(phase_id)
                            if instance and instance.latency_us > 0 and hasattr(phase, 'traffic') and phase.traffic:
                                weight_loading_traffic_rows.append({
                                    "Input": int(input_idx),
                                    "Op ID": int(phase_id),
                                    "Traffic": str(phase.traffic)
                                })
            
            if weight_loading_traffic_rows:
                children.append(MetricNode(
                    section=section,
                    name="Weight Loading Traffic Table",
                    columns=["Input", "Op ID", "Traffic"],
                    rows=weight_loading_traffic_rows
                ))
            else:
                children.append(MetricNode(
                    section=section,
                    name="Weight Loading Traffic Table",
                    description="No weight loading traffic data available."
                ))
        else:
            children.append(MetricNode(
                section=section,
                name="Weight Loading Traffic Table",
                description="Weight loading traffic data not available."
            ))
            
        # ================================================================
        # Section: Layer traffic (bytes)
        # ================================================================
        section = "Activation Communication Traffic (bytes)"
        activation_comm_traffic_dict = getattr(mapped_model, 'layer_activation_comm_traffic', None)
        activation_comm_traffic_table_rows = []
        if activation_comm_traffic_dict is not None:
            for layer_idx in sorted(activation_comm_traffic_dict.keys()):
                traffic_data = activation_comm_traffic_dict[layer_idx]
                row = {
                    "Layer": int(layer_idx),
                    "Traffic": str(traffic_data)
                }
                activation_comm_traffic_table_rows.append(row)
            if activation_comm_traffic_table_rows:
                children.append(MetricNode(
                    section=section,
                    name="Activation Communication Traffic Table",
                    columns=["Layer", "Traffic"],
                    rows=activation_comm_traffic_table_rows
                ))
            else:
                children.append(MetricNode(
                    section=section,
                    name="Activation Communication Traffic Table",
                    description="Activation communication traffic data not available."
                ))
        else:
            children.append(MetricNode(
                section=section,
                name="Activation Communication Traffic Table",
                description="Activation communication traffic data not available."
            ))

        # ================================================================
        # Section: Communication Co-active Layers
        # ================================================================
        section = "Communication Co-active Layers"
        co_active_traffic_dict = getattr(mapped_model, 'communication_co_active_traffic', {})
        co_active_layers_table_rows = []
        
        # Extract co-active layer information from co-active traffic data
        for sim_call, data in co_active_traffic_dict.items():
            phases_simulated_together = data.get('phases_simulated_together', [])
            if len(phases_simulated_together) > 1:  # Only when multiple phases are active together
                for phase_data in phases_simulated_together:
                    phase_model_idx = phase_data.get('model_idx', -1)
                    input_idx = phase_data.get('input_idx', -1)
                    layer_idx = phase_data.get('layer_idx', -1)
                    phase_type = phase_data.get('phase_type', 'Unknown')
                    
                    # Find other phases that were active at the same time
                    co_active_with = []
                    for other_phase in phases_simulated_together:
                        if other_phase != phase_data:  # Don't include self
                            other_model_idx = other_phase.get('model_idx', -1)
                            other_input_idx = other_phase.get('input_idx', -1)
                            other_layer_idx = other_phase.get('layer_idx', -1)
                            other_phase_type = other_phase.get('phase_type', 'Unknown')
                            
                            # Get model name for other phase
                            other_model_name = "Unknown"
                            if other_model_idx in retired_models:
                                other_model_name = getattr(retired_models[other_model_idx], 'model_name', 'Unknown')
                            
                            co_active_with.append(f"Model {other_model_idx} ({other_model_name}), Input {other_input_idx}, Layer {other_layer_idx} ({other_phase_type})")
                    
                    if co_active_with:
                        row = {
                            "Model": int(phase_model_idx),
                            "Input": int(input_idx),
                            "Layer": int(layer_idx) if layer_idx != -1 else "N/A",
                            "Phase Type": phase_type,
                            "Co-Active With": "; ".join(co_active_with)
                        }
                        co_active_layers_table_rows.append(row)
        
        if co_active_layers_table_rows:
            children.append(MetricNode(
                section=section,
                name="Co-Active Layers Table",
                columns=["Model", "Input", "Layer", "Phase Type", "Co-Active With"],
                rows=co_active_layers_table_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Co-Active Layers Table",
                description="No co-active layers detected for this model"
            ))
            
        # ================================================================
        # Section: Communication Co-active Traffic
        # ================================================================
        section = "Communication Co-active Traffic"
        co_active_traffic_dict = getattr(mapped_model, 'communication_co_active_traffic', {})
        co_active_traffic_table_rows = []
        
        if co_active_traffic_dict:
            for simulation_call_id in sorted(co_active_traffic_dict.keys()):
                call_info = co_active_traffic_dict[simulation_call_id]
                simulation_time = call_info['simulation_time_us']
                phases_info = call_info['phases_simulated_together']
                
                # Create traffic breakdown by source (model, input, layer) for all phases
                all_phases_traffic_by_source = {}
                for phase_info in phases_info:
                    traffic_contribution = phase_info['traffic_contribution']
                    if traffic_contribution:
                        phase_model_idx = phase_info['model_idx']
                        input_idx = phase_info['input_idx']
                        layer_idx = phase_info['layer_idx']
                        phase_type = phase_info['phase_type']
                        phase_id = phase_info['phase_id']
                        
                        # Create source identifier
                        if phase_type == 'ACTIVATION_COMM':
                            source_key = f"Model {phase_model_idx}, Input {input_idx}, Layer {layer_idx}"
                        elif phase_type == 'WEIGHT_LOADING_COMM':
                            next_compute_layer = phase_info['next_compute_layer']
                            if next_compute_layer is not None and next_compute_layer >= 0:
                                source_key = f"Model {phase_model_idx}, Input {input_idx}, Weight Loading for Layer {next_compute_layer}"
                            else:
                                source_key = f"Model {phase_model_idx}, Input {input_idx}, Weight Loading"
                        else:
                            source_key = f"Model {phase_model_idx}, Input {input_idx}, Layer {layer_idx}"
                        
                        all_phases_traffic_by_source[source_key] = {
                            'traffic': traffic_contribution,
                            'phase_id': phase_id
                        }
                
                # Create a row for each phase in this simulation call
                for phase_info in phases_info:
                    phase_model_idx = phase_info['model_idx']
                    input_idx = phase_info['input_idx']
                    phase_id = phase_info['phase_id']
                    phase_type = phase_info['phase_type']
                    layer_idx = phase_info['layer_idx']
                    next_compute_layer = phase_info['next_compute_layer']
                    
                    # Create co-active traffic excluding the current phase
                    co_active_traffic_by_source = {}
                    for source_key, source_data in all_phases_traffic_by_source.items():
                        if source_data['phase_id'] != phase_id:  # Exclude current phase
                            co_active_traffic_by_source[source_key] = source_data['traffic']
                    
                    # Format co-active traffic by source as string
                    co_active_traffic_str = str(co_active_traffic_by_source) if co_active_traffic_by_source else "{}"
                    
                    # Determine layer information for display
                    if phase_type == 'ACTIVATION_COMM':
                        layer_display = layer_idx if layer_idx >= 0 else "N/A"
                        next_layer_display = ""
                    elif phase_type == 'WEIGHT_LOADING_COMM':
                        if next_compute_layer is not None and next_compute_layer >= 0:
                            layer_display = next_compute_layer
                            next_layer_display = f"Next: Layer {next_compute_layer}"
                        else:
                            layer_display = "N/A"
                            next_layer_display = ""
                    else:
                        layer_display = layer_idx if layer_idx >= 0 else "N/A"
                        next_layer_display = ""
                    
                    # Convert phase type to user-friendly format
                    if phase_type == 'ACTIVATION_COMM':
                        phase_type_display = 'Activation Communication'
                    elif phase_type == 'WEIGHT_LOADING_COMM':
                        phase_type_display = 'Weight Loading'
                    else:
                        phase_type_display = phase_type
                    
                    row = {
                        "Sim Call": int(simulation_call_id),
                        "Time (μs)": simulation_time,
                        "Model": int(phase_model_idx),
                        "Input": int(input_idx),
                        "Layer": layer_display,
                        "Phase": int(phase_id),
                        "Type": phase_type_display,
                        "Next Layer": next_layer_display,
                        "Co-active Traffic": co_active_traffic_str
                    }
                    co_active_traffic_table_rows.append(row)
        
        if co_active_traffic_table_rows:
            children.append(MetricNode(
                section=section,
                name="Co-active Traffic Table",
                columns=["Sim Call", "Time (μs)", "Model", "Input", "Layer", "Phase", "Type", "Next Layer", "Co-active Traffic"],
                rows=co_active_traffic_table_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Co-active Traffic Table",
                description="No communication co-active traffic data available."
            ))
            
        # ================================================================
        # Section: Layer Energy per input
        # ================================================================
        section = "Layer Energy per input"
        compute_energy_dict = getattr(mapped_model, 'layer_compute_energy', {})
        energy_table_rows = []
        for input_idx in range(num_inputs):
            layers_for_input = set()
            if input_idx in compute_energy_dict: layers_for_input.update(compute_energy_dict[input_idx].keys())
            for layer_idx in sorted(list(layers_for_input)):
                row = {
                    "Input": int(input_idx),
                    "Layer": int(layer_idx),
                    "Compute Energy (fJ)": compute_energy_dict.get(input_idx, {}).get(layer_idx, -1.0),
                }
                energy_table_rows.append(row)
        if energy_table_rows:
            children.append(MetricNode(
                section=section,
                name="Energy Table",
                columns=["Input", "Layer", "Compute Energy (fJ)"],
                rows=energy_table_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Energy Table",
                description="No layer energy data found for any input."
            ))
            
        # ================================================================
        # Section: Model Mapping
        # ================================================================
        section = "Model Mapping"
        mapping_data = getattr(mapped_model, 'mapping', [])
        total_chiplets = 0
        if mapping_data:
            all_mapped_chiplets = set()
            for _, mappings in mapping_data:
                for chiplet_id, _ in mappings:
                    all_mapped_chiplets.add(chiplet_id)
            total_chiplets = len(all_mapped_chiplets)
        children.append(MetricNode(section=section, name="Total chiplets used", value=int(total_chiplets), unit=""))
        mapping_table_rows = []
        for layer_idx, mappings in sorted(mapping_data, key=lambda item: item[0]) if mapping_data else []:
            for chiplet_id, percentage in sorted(mappings, key=lambda item: item[0]):
                row = {
                    "Layer": int(layer_idx),
                    "Chiplet": int(chiplet_id),
                    "Percent": percentage
                }
                mapping_table_rows.append(row)
        if mapping_table_rows:
            children.append(MetricNode(
                section=section,
                name="Mapping Table",
                columns=["Layer", "Chiplet", "Percent"],
                rows=mapping_table_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Mapping Table",
                description="No mapping information available."
            ))
            
        model_name = f"Model {model_idx}: {mapped_model.model_name}"
        return MetricNode(section="Model", name=model_name, children=children)

    def format_utilization_metrics(self, time_step_us: float) -> MetricNode:
        """
        Format system utilization metrics as a MetricNode tree.
        Args:
            time_step_us (float): Time step used for the analysis
        Returns:
            MetricNode: Root node for utilization metrics
        """
        mc = self.mc
        
        # Comprehensive check for all required metrics
        required_method1_compute = (\
            mc.per_chiplet_compute_time_utilization_pct is None or\
            mc.chiplet_total_compute_time_us is None or\
            mc.mean_compute_time_chiplet_utilization_pct is None\
        )
        required_method1_activation_comm = (\
            mc.per_chiplet_activation_comm_time_utilization_pct is None or\
            mc.chiplet_total_activation_comm_time_us is None or\
            mc.mean_activation_comm_time_chiplet_utilization_pct is None\
        )
        required_method1_weight_loading = (\
            mc.per_chiplet_weight_loading_time_utilization_pct is None or\
            mc.chiplet_total_weight_loading_time_us is None or\
            mc.mean_weight_loading_time_chiplet_utilization_pct is None\
        )
        required_method1_combined = (\
            mc.per_chiplet_combined_time_utilization_pct is None or\
            mc.chiplet_total_combined_busy_time_us is None or\
            mc.mean_combined_time_chiplet_utilization_pct is None\
        )

        # Method 2 metrics for activity/utilization over time (not averages)
        required_method2_activity_compute = (
            mc.chiplet_compute_activity_over_time is None or
            mc.system_compute_utilization_over_time_pct is None
        )
        required_method2_activity_activation_comm = (
            mc.chiplet_activation_comm_activity_over_time is None or
            mc.system_activation_comm_utilization_over_time_pct is None
        )
        required_method2_activity_weight_loading = (
            mc.chiplet_weight_loading_activity_over_time is None or
            mc.system_weight_loading_utilization_over_time_pct is None
        )
        required_method2_activity_combined = (
            mc.chiplet_combined_activity_over_time is None or
            mc.system_combined_utilization_over_time_pct is None
        )

        if (required_method1_compute or required_method1_activation_comm or required_method1_weight_loading or required_method1_combined or
            required_method2_activity_compute or required_method2_activity_activation_comm or required_method2_activity_weight_loading or required_method2_activity_combined):
            return MetricNode(section="Utilization", name="Utilization", description="Not all utilization metrics are available for formatting.")
        
        # Create children nodes for all metrics
        children = []
        
        # ================================================================
        # Section: System-Wide Metrics
        # ================================================================
        children.append(MetricNode(
            section="System-Wide Metrics",
            name="Time Step Used for Analysis",
            value=time_step_us,
            unit="μs"
        ))

        # --- Method 1: Mean of Per-Chiplet Total Time Based Utilizations ---
        children.append(MetricNode(
            section="System-Wide Metrics",
            name="Mean Per-Chiplet Compute Time Utilization (Method 1)",
            value=mc.mean_compute_time_chiplet_utilization_pct,
            unit="%"
        ))
        children.append(MetricNode(
            section="System-Wide Metrics",
            name="Mean Per-Chiplet Activation Communication Time Utilization (Method 1)",
            value=mc.mean_activation_comm_time_chiplet_utilization_pct,
            unit="%"
        ))
        children.append(MetricNode(
            section="System-Wide Metrics",
            name="Mean Per-Chiplet Weight Loading Time Utilization (Method 1)",
            value=mc.mean_weight_loading_time_chiplet_utilization_pct,
            unit="%"
        ))
        children.append(MetricNode(
            section="System-Wide Metrics",
            name="Mean Per-Chiplet Combined Time Utilization (Method 1)",
            value=mc.mean_combined_time_chiplet_utilization_pct,
            unit="%"
        ))
        
        # ================================================================
        # Section: Total Time Based Utilization Tables (Method 1)
        # ================================================================
        # Compute (Method 1)
        total_compute_rows_m1 = []
        for chiplet_id in sorted(mc.per_chiplet_compute_time_utilization_pct.keys()):
            compute_time = mc.chiplet_total_compute_time_us.get(chiplet_id, 0.0)
            utilization = mc.per_chiplet_compute_time_utilization_pct.get(chiplet_id, 0.0)
            total_compute_rows_m1.append({
                "Chiplet ID": chiplet_id,
                "Total Compute Time (μs)": compute_time,
                "Utilization (%)": utilization
            })
        children.append(MetricNode(
            section="Total Compute Utilization (Method 1)",
            name="Compute Time Based Utilization (Per Chiplet)",
            columns=["Chiplet ID", "Total Compute Time (μs)", "Utilization (%)"],
            rows=total_compute_rows_m1
        ))

        # Activation Communication (Method 1)
        total_activation_comm_rows_m1 = []
        for chiplet_id in sorted(mc.per_chiplet_activation_comm_time_utilization_pct.keys()):
            activation_comm_time = mc.chiplet_total_activation_comm_time_us.get(chiplet_id, 0.0)
            utilization = mc.per_chiplet_activation_comm_time_utilization_pct.get(chiplet_id, 0.0)
            total_activation_comm_rows_m1.append({
                "Chiplet ID": chiplet_id,
                "Total Activation Communication Time (μs)": activation_comm_time,
                "Utilization (%)": utilization
            })
        children.append(MetricNode(
            section="Total Activation Communication Utilization (Method 1)",
            name="Activation Communication Time Based Utilization (Per Chiplet)",
            columns=["Chiplet ID", "Total Activation Communication Time (μs)", "Utilization (%)"],
            rows=total_activation_comm_rows_m1
        ))

        # Weight Loading (Method 1)
        total_weight_loading_rows_m1 = []
        for chiplet_id in sorted(mc.per_chiplet_weight_loading_time_utilization_pct.keys()):
            weight_loading_time = mc.chiplet_total_weight_loading_time_us.get(chiplet_id, 0.0)
            utilization = mc.per_chiplet_weight_loading_time_utilization_pct.get(chiplet_id, 0.0)
            total_weight_loading_rows_m1.append({
                "Chiplet ID": chiplet_id,
                "Total Weight Loading Time (μs)": weight_loading_time,
                "Utilization (%)": utilization
            })
        children.append(MetricNode(
            section="Total Weight Loading Utilization (Method 1)",
            name="Weight Loading Time Based Utilization (Per Chiplet)",
            columns=["Chiplet ID", "Total Weight Loading Time (μs)", "Utilization (%)"],
            rows=total_weight_loading_rows_m1
        ))

        # Combined (Method 1) - Compute + Activation Communication + Weight Loading
        total_combined_rows_m1 = []
        for chiplet_id in sorted(mc.per_chiplet_combined_time_utilization_pct.keys()):
            combined_time = mc.chiplet_total_combined_busy_time_us.get(chiplet_id, 0.0)
            utilization = mc.per_chiplet_combined_time_utilization_pct.get(chiplet_id, 0.0)
            total_combined_rows_m1.append({
                "Chiplet ID": chiplet_id,
                "Total Combined Busy Time (μs)": combined_time,
                "Utilization (%)": utilization
            })
        children.append(MetricNode(
            section="Total Combined Utilization (Method 1)",
            name="Combined Busy Time Based Utilization (Per Chiplet) - Compute + Activation Communication + Weight Loading",
            columns=["Chiplet ID", "Total Combined Busy Time (μs)", "Utilization (%)"],
            rows=total_combined_rows_m1
        ))
        
        # ================================================================
        # Return a single MetricNode with children that should not be saved separately
        # ================================================================
        return MetricNode(
            section="Utilization", 
            name="System Utilization Metrics", 
            children=children,
            save_children_separately=False,  # Don't save children as separate files
            children_folder_name="formatted_utilization_metrics"  # Use this folder to save the metrics
        )
        
    def format_approach_comparison_metrics(self) -> MetricNode:
        """
        Formats the approach comparison metrics into a MetricNode, creating separate tables
        for total, compute, and communication latencies, and their respective percent differences.

        Returns:
            MetricNode: A root node containing the formatted approach comparison metrics.
        """
        mc = self.mc
        children = []
        
        # Check if the necessary data is available
        if not mc.approach_comparison_metrics or \
           "model_type_performance_comparison" not in mc.approach_comparison_metrics:
            return MetricNode(
                section="Approach Comparison",
                name="Approach Comparison Metrics",
                description="Approach comparison metrics not available or not computed."
            )

        comparison_data = mc.approach_comparison_metrics.get("model_type_performance_comparison", {})

        if not comparison_data:
            children.append(MetricNode(
                section="Approach Comparison", # Changed section name to be more general for the overall group
                name="Latency Comparison Data", # Changed name to be more general for this specific message
                description="No data available for approach comparison."
            ))
        else:
            all_model_names = sorted(comparison_data.keys())

            # Helper function for percent difference calculation
            def calculate_percent_difference(value, base):
                if base is not None:
                    if base > 0:
                        if value is not None:
                            if value == base: return "0.00%"
                            return f"{((value - base) / base) * 100:.2f}%"
                    elif base == 0: # Base is zero
                        if value is not None:
                            if value == 0: return "0.00%"
                            return "Inf" # Division by zero, value is non-zero
                return "N/A" # Base or value is None

            # --- Table 1: Average Total Latency Comparison ---
            total_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_metrics = data.get("main_simulation", {})
                individual_model_metrics = data.get("individual_model", {})
                empty_system_model_metrics = data.get("empty_system_model", {})
                row = {
                    "Model Type": model_type_name,
                    "Main Sim Avg Total Latency (µs)": main_sim_metrics.get("avg_time_per_input"),
                    "Individual Model Avg Latency (µs)": individual_model_metrics.get("avg_time_per_input"),
                    "Empty System Model Avg Latency (µs)": empty_system_model_metrics.get("avg_time_per_input")
                }
                total_latency_rows.append(row)
            
            if total_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Latency Values",
                    name="Average Total Latency Comparison by Model Type",
                    columns=[
                        "Model Type", 
                        "Main Sim Avg Total Latency (µs)",
                        "Individual Model Avg Latency (µs)", 
                        "Empty System Model Avg Latency (µs)"
                    ],
                    rows=total_latency_rows,
                    description="Compares average per-input total execution times for each model type across different simulation/modeling approaches."
                ))

            # --- Table 2: Average Compute Latency Comparison ---
            compute_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_metrics = data.get("main_simulation", {})
                empty_system_model_metrics = data.get("empty_system_model", {})
                row = {
                    "Model Type": model_type_name,
                    "Main Sim Avg Compute Latency (µs)": main_sim_metrics.get("avg_compute_time_per_input"),
                    "Empty System Model Avg Compute Latency (µs)": empty_system_model_metrics.get("avg_compute_time_per_input")
                }
                compute_latency_rows.append(row)

            if compute_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Latency Values",
                    name="Average Compute Latency Comparison by Model Type",
                    columns=[
                        "Model Type",
                        "Main Sim Avg Compute Latency (µs)",
                        "Empty System Model Avg Compute Latency (µs)"
                    ],
                    rows=compute_latency_rows,
                    description="Compares average per-input compute times for each model type between Main Simulation and Empty System Model."
                ))

            # --- Table 3: Average Communication Latency Comparison ---
            comm_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_metrics = data.get("main_simulation", {})
                empty_system_model_metrics = data.get("empty_system_model", {})
                row = {
                    "Model Type": model_type_name,
                    "Main Sim Avg Comm Latency (µs)": main_sim_metrics.get("avg_comm_time_per_input"),
                    "Empty System Model Avg Comm Latency (µs)": empty_system_model_metrics.get("avg_comm_time_per_input")
                }
                comm_latency_rows.append(row)

            if comm_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Latency Values",
                    name="Average Communication Latency Comparison by Model Type",
                    columns=[
                        "Model Type",
                        "Main Sim Avg Comm Latency (µs)",
                        "Empty System Model Avg Comm Latency (µs)"
                    ],
                    rows=comm_latency_rows,
                    description="Compares average per-input communication times for each model type between Main Simulation and Empty System Model."
                ))

            # --- Percent Difference Tables ---

            # --- Percent Diff Table 1: Total Latency vs. Empty System ---
            pd_total_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_total = data.get("main_simulation", {}).get("avg_time_per_input")
                individual_total = data.get("individual_model", {}).get("avg_time_per_input")
                empty_total = data.get("empty_system_model", {}).get("avg_time_per_input")
                
                pd_row = {
                    "Model Type": model_type_name,
                    "Main Sim vs. Empty System (%)": calculate_percent_difference(main_sim_total, empty_total),
                    "Individual Model vs. Empty System (%)": calculate_percent_difference(individual_total, empty_total)
                }
                pd_total_latency_rows.append(pd_row)
            
            if pd_total_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Percent Differences",
                    name="Total Latency: Percent Difference vs. Empty System Model",
                    columns=["Model Type", "Main Sim vs. Empty System (%)", "Individual Model vs. Empty System (%)"],
                    rows=pd_total_latency_rows,
                    description="Percentage difference of 'Main Sim' and 'Individual Model' average total latencies compared to 'Empty System Model'."
                ))

            # --- Percent Diff Table 2: Compute Latency vs. Empty System ---
            pd_compute_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_compute = data.get("main_simulation", {}).get("avg_compute_time_per_input")
                empty_compute = data.get("empty_system_model", {}).get("avg_compute_time_per_input")
                
                pd_row = {
                    "Model Type": model_type_name,
                    "Main Sim vs. Empty System (%)": calculate_percent_difference(main_sim_compute, empty_compute)
                }
                pd_compute_latency_rows.append(pd_row)

            if pd_compute_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Percent Differences",
                    name="Compute Latency: Percent Difference vs. Empty System Model",
                    columns=["Model Type", "Main Sim vs. Empty System (%)"],
                    rows=pd_compute_latency_rows,
                    description="Percentage difference of 'Main Sim' average compute latency compared to 'Empty System Model'."
                ))

            # --- Percent Diff Table 3: Communication Latency vs. Empty System ---
            pd_comm_latency_rows = []
            for model_type_name in all_model_names:
                data = comparison_data.get(model_type_name, {})
                main_sim_comm = data.get("main_simulation", {}).get("avg_comm_time_per_input")
                empty_comm = data.get("empty_system_model", {}).get("avg_comm_time_per_input")
                
                pd_row = {
                    "Model Type": model_type_name,
                    "Main Sim vs. Empty System (%)": calculate_percent_difference(main_sim_comm, empty_comm)
                }
                pd_comm_latency_rows.append(pd_row)

            if pd_comm_latency_rows:
                children.append(MetricNode(
                    section="Approach Comparison: Percent Differences",
                    name="Communication Latency: Percent Difference vs. Empty System Model",
                    columns=["Model Type", "Main Sim vs. Empty System (%)"],
                    rows=pd_comm_latency_rows,
                    description="Percentage difference of 'Main Sim' average communication latency compared to 'Empty System Model'."
                ))
            
            if not children: # Fallback if all_model_names was empty or no data processed
                 children.append(MetricNode(
                    section="Approach Comparison",
                    name="Latency Comparison Data",
                    description="No data processed for approach comparison tables."
                ))

        return MetricNode(
            section="Approach Comparison",
            name="Approach Comparison Metrics",
            children=children,
            save_children_separately=False, # Main node for approach comparison
            children_folder_name="formatted_approach_comparison_metrics" # Subfolder for any potential future individual saves from this group
        )
    
    def format_simulation_summary(self, main_simulation_duration_seconds: float = 0.0) -> MetricNode:
        """
        Creates a simulation summary MetricNode with information about the simulation setup and results.
        This includes workload details, system configuration, and basic timing information.
        
        Args:
            main_simulation_duration_seconds (float): Duration of the main simulation in seconds
            
        Returns:
            MetricNode: Formatted simulation summary
        """
        gm = self.gm
        
        # Get basic workload and system information
        workload_file = os.path.basename(gm.workload_manager.wl_file) if hasattr(gm.workload_manager, 'wl_file') else "Unknown"
        adj_matrix_file = os.path.basename(gm.adj_matrix_file) if hasattr(gm, 'adj_matrix_file') else "Unknown"
        chiplet_mapping_file = os.path.basename(gm.chiplet_mapping_file) if hasattr(gm, 'chiplet_mapping_file') else "Unknown"
        communication_simulator = gm.communication_simulator if hasattr(gm, 'communication_simulator') else "Unknown"
        communication_method = gm.communication_method if hasattr(gm, 'communication_method') else "Unknown"
        mapping_function = gm.mapping_function if hasattr(gm, 'mapping_function') else "Unknown"
        
        # Calculate total simulation time in microseconds
        # Get the max model completion time among all retired models
        simulation_runtime_us = 0.0
        if hasattr(gm, 'post_warmup_retired_models') and gm.post_warmup_retired_models:
            max_completion_time = max(
                model.model_completion_time_us 
                for model in gm.post_warmup_retired_models.values() 
                if model.model_completion_time_us > 0
            )
            simulation_runtime_us = max_completion_time
        
        # Count total models processed
        total_models = len(gm.post_warmup_retired_models) if hasattr(gm, 'post_warmup_retired_models') else 0
        
        # Create children nodes
        children = []
        
        # ================================================================
        # Section: Configuration
        # ================================================================
        section = "Configuration"
        children.append(MetricNode(
            section=section,
            name="Workload",
            value=workload_file
        ))
        
        children.append(MetricNode(
            section=section,
            name="Adjacency Matrix",
            value=adj_matrix_file
        ))
        
        children.append(MetricNode(
            section=section,
            name="Chiplet Mapping",
            value=chiplet_mapping_file
        ))
        
        children.append(MetricNode(
            section=section,
            name="Communication Simulator",
            value=communication_simulator
        ))
        
        children.append(MetricNode(
            section=section,
            name="Communication Method",
            value=communication_method
        ))
        
        children.append(MetricNode(
            section=section,
            name="Mapping Function",
            value=mapping_function
        ))
        
        # ================================================================
        # Section: Timing
        # ================================================================
        section = "Timing"
        children.append(MetricNode(
            section=section,
            name="Simulation Wall Clock Time",
            value=main_simulation_duration_seconds,
            unit="seconds"
        ))
        
        children.append(MetricNode(
            section=section,
            name="Total Models Processed",
            value=total_models
        ))
        
        children.append(MetricNode(
            section=section,
            name="Simulation Runtime",
            value=simulation_runtime_us,
            unit="μs"
        ))
        
        # ================================================================
        # Section: Models
        # ================================================================
        section = "Models"
        
        # Create a table of models processed in order of start time
        model_rows = []
        if hasattr(gm, 'post_warmup_retired_models'):
            # Sort models by start time
            sorted_models = sorted(
                gm.post_warmup_retired_models.items(),
                key=lambda x: x[1].model_start_time_us
            )
            
            for model_idx, model in sorted_models:
                model_rows.append({
                    "Model ID": int(model_idx),
                    "Name": model.model_name,
                    "Start Time (μs)": model.model_start_time_us
                })
        
        if model_rows:
            children.append(MetricNode(
                section=section,
                name="Models By Start Time",
                columns=["Model ID", "Name", "Start Time (μs)"],
                rows=model_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Models By Start Time",
                description="No models were processed during the simulation."
            ))
        
        # Create a table of models processed in order of completion time
        completion_time_rows = []
        if hasattr(gm, 'post_warmup_retired_models'):
            # Sort models by completion time
            sorted_models_by_completion = sorted(
                gm.post_warmup_retired_models.items(),
                key=lambda x: x[1].model_completion_time_us
            )
            
            for model_idx, model in sorted_models_by_completion:
                completion_time_rows.append({
                    "Model ID": int(model_idx),
                    "Name": model.model_name,
                    "Completion Time (μs)": model.model_completion_time_us,
                    "Duration (μs)": model.model_completion_time_us - model.model_start_time_us
                })
        
        if completion_time_rows:
            children.append(MetricNode(
                section=section,
                name="Models by Completion Time",
                columns=["Model ID", "Name", "Completion Time (μs)", "Duration (μs)"],
                rows=completion_time_rows
            ))
        else:
            children.append(MetricNode(
                section=section,
                name="Models by Completion Time",
                description="No models were processed during the simulation."
            ))
        
        # Return a root MetricNode containing all the sections
        return MetricNode(
            section="Simulation Summary",
            name="Simulation Summary",
            children=children,
            save_children_separately=False
        )

    def format_energy_metrics(self) -> MetricNode:
        """
        Formats energy metrics into a MetricNode tree.
        This includes system-wide and per-chiplet energy summaries.
        
        Returns:
            MetricNode: A root node containing the formatted energy metrics.
        """
        mc = self.mc
        children = []

        # Check for availability of metrics
        required_metrics = [
            'system_total_energy_uj', 'system_compute_energy_uj', 'system_communication_energy_uj',
            'chiplet_total_energy_uj', 'chiplet_compute_energy_uj', 'chiplet_communication_energy_uj',
        ]
        
        for metric_name in required_metrics:
            if getattr(mc, metric_name, None) is None:
                return MetricNode(
                    section="Energy",
                    name="Energy Metrics",
                    description=f"Energy metrics not available. Metric '{metric_name}' is missing."
                )

        # --- Section: System-Wide Energy Summary ---
        section_system_energy = "System-Wide Energy Summary"
        system_energy_children = [
            MetricNode(
                section=section_system_energy,
                name="Total System Energy",
                value=mc.system_total_energy_uj,
                unit="μJ"
            ),
            MetricNode(
                section=section_system_energy,
                name="Total System Compute Energy",
                value=mc.system_compute_energy_uj,
                unit="μJ"
            ),
            MetricNode(
                section=section_system_energy,
                name="Total System Communication Energy",
                value=mc.system_communication_energy_uj,
                unit="μJ"
            )
        ]
        children.append(MetricNode(section=section_system_energy, name=section_system_energy, children=system_energy_children))


        # --- Section: Per-Chiplet Energy Summary ---
        section_chiplet_energy = "Per-Chiplet Energy Summary"
        all_chiplet_ids = set(mc.chiplet_compute_energy_uj.keys()) | set(mc.chiplet_communication_energy_uj.keys()) | set(mc.chiplet_total_energy_uj.keys())
        
        if all_chiplet_ids:
            energy_rows = []
            for cid in sorted(list(all_chiplet_ids)):
                row = {
                    "Chiplet ID": cid,
                    "Compute Energy (μJ)": mc.chiplet_compute_energy_uj.get(cid, 0.0),
                    "Communication Energy (μJ)": mc.chiplet_communication_energy_uj.get(cid, 0.0),
                    "Total Energy (μJ)": mc.chiplet_total_energy_uj.get(cid, 0.0),
                }
                energy_rows.append(row)
            
            children.append(MetricNode(
                section=section_chiplet_energy,
                name="Per-Chiplet Energy Consumption",
                columns=["Chiplet ID", "Compute Energy (μJ)", "Communication Energy (μJ)", "Total Energy (μJ)"],
                rows=energy_rows
            ))
        else:
            children.append(MetricNode(
                section=section_chiplet_energy,
                name="Per-Chiplet Energy Consumption",
                description="No per-chiplet energy data available."
            ))

        return MetricNode(
            section="Energy",
            name="Energy Metrics",
            children=children,
            save_children_separately=True,
            children_folder_name="formatted_energy_metrics"
        )
    
    
    