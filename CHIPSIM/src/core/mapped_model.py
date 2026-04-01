from typing import Dict, List, Optional
import os
import logging

from .comm_types import (
    Phase, PhaseInstance, PhaseState, TrafficMatrixDict,
    ComputePhase, ActivationCommPhase, WeightLoadingPhase,
    create_compute_phase, create_activation_comm_phase, create_weight_loading_phase
)
from .traffic_calculator import TrafficCalculator


class MappedModel:
    def __init__(self, 
                 model_name, 
                 mapping,
                 model_start_time_us,
                 model_def,
                 model_metrics,
                 num_inputs,
                 used_capacity,
                 model_idx,
                 communication_method,
                 traffic_calculator: TrafficCalculator,
                 weight_stationary: bool = True):
        
        # === Basic Model Identification ===
        self.model_name = model_name
        self.model_idx = model_idx
        
        # === Model Configuration & Definition ===
        self.model_def = model_def         # Original model definition (expects {'layers': {idx: {...}}})
        self.model_metrics = model_metrics # Pre-calculated metrics per layer
        self.mapping = mapping                 # Layer-to-chiplet mapping
        self.num_inputs = num_inputs           # Number of concurrent inputs the model processes
        # New format: number of layers comes from length of model_def['layers']
        if 'layers' not in self.model_def or not isinstance(self.model_def['layers'], dict):
            raise ValueError("Model definition must contain a 'layers' dictionary in the new format.")
        self.num_layers = len(self.model_def['layers']) # Total layers
        self.communication_method = communication_method # Communication method ('pipelined' or 'non-pipelined')
        self.weight_stationary = weight_stationary # Weight stationary mode for in-memory computing
        
        # === Timing ===
        self.model_start_time_us = model_start_time_us      # Global time when the first input starts processing
        
        # Initialize input start times. Only input 0 starts at model_start_time_us.
        # Subsequent input start times are calculated dynamically based on communication_method.
        self.input_start_time_us = [-1] * self.num_inputs
        if self.num_inputs > 0:
             self.input_start_time_us[0] = self.model_start_time_us
        
        # Initialize completion times to -1 (to be calculated later)
        self.input_completion_time_us = [-1] * self.num_inputs
        self.model_completion_time_us = -1
        
        # === Physical Resource Usage ===
        self.used_capacity = used_capacity   # Capacity units allocated to this model (IMC: crossbars, CMOS: weights)
        self.layer_chiplets = {}               # Chiplets used by each layer {layer_idx: {chiplet_id, ...}}
        self.chiplet_crossbars_occupied = {}   # Crossbars occupied per chiplet {chiplet_id: count}
        
        # === Layer Chunk Data (Input for Compute Simulation) ===
        self.layer_chunks = {}                 # Chunks for each layer {layer_idx: [chunk_data, ...]}
        
        # === Simulation Results - Per Chunk ===
        self.chunk_results = {}                # Raw results per chunk {layer_idx: {chiplet_id: (chunk_idx, result)}}
        
        # === NEW: Flexible Phase System ===
        # Phase Definition (shared across all inputs)
        self.phases: Dict[int, Phase] = {}  # {phase_id: Phase}
        self.phase_execution_order: List[int] = []  # Linear execution order
        self.next_phase_id = 0  # Counter for generating phase IDs
        
        # Phase Instances (per input)
        self.phase_instances: Dict[int, Dict[int, PhaseInstance]] = {
            input_idx: {} for input_idx in range(self.num_inputs)
        }  # {input_idx: {phase_id: PhaseInstance}}
        
        # Quick lookup mappings
        self.layer_to_compute_phase: Dict[int, int] = {}  # {layer_idx: compute_phase_id}
        self.layer_to_activation_comm_phase: Dict[int, int] = {}  # {layer_idx: comm_phase_id}
        self.layer_to_weight_comm_phase: Dict[int, int] = {}  # {layer_idx: weight_phase_id}
        
        # === Traffic Calculator ===
        self.traffic_calculator = traffic_calculator
        
        # === Weight Loading Tracking (for weight stationary mode) ===
        # Track which layers have had their weights loaded (model-specific)
        self.weights_loaded: Dict[int, bool] = {i: False for i in range(self.num_layers)}
        
        # === Layer-Based Metrics (populated from phase data via sync_phase_metrics_to_layers()) ===
        # Compute Metrics
        self.layer_compute_latency = {input_idx: {} for input_idx in range(self.num_inputs)}
        self.layer_compute_energy = {input_idx: {} for input_idx in range(self.num_inputs)}
        
        # Phase-Type-Specific Communication Metrics
        # Activation Communication Metrics
        self.layer_activation_comm_latency = {input_idx: {} for input_idx in range(self.num_inputs)}
        self.layer_activation_comm_traffic = {}  # {layer_idx: traffic_matrix}
        
        # Weight Loading Communication Metrics
        self.layer_weight_loading_latency = {input_idx: {} for input_idx in range(self.num_inputs)}
        self.layer_weight_loading_traffic = {}  # {layer_idx: traffic_matrix}
        
        # Layer timing (earliest start, latest completion across all phases for a layer)
        self.layer_start_time_us = {input_idx: {} for input_idx in range(self.num_inputs)}
        self.layer_completion_time_us = {input_idx: {} for input_idx in range(self.num_inputs)}
        
        # Phase-Type-Specific Communication Tracking (for debugging/analysis)
        # Activation Communication Tracking
        self.layer_activation_comm_latency_history = {
            input_idx: {} for input_idx in range(self.num_inputs)
        }
        self.layer_activation_comm_is_active = {
            input_idx: {} for input_idx in range(self.num_inputs)
        }
        
        # Weight Loading Communication Tracking
        self.layer_weight_loading_latency_history = {
            input_idx: {} for input_idx in range(self.num_inputs)
        }
        self.layer_weight_loading_is_active = {
            input_idx: {} for input_idx in range(self.num_inputs)
        }
        
        # === Communication Co-active Traffic Tracking ===
        # Track what other communication phases were active during each simulation call
        self.communication_co_active_traffic = {}  # {simulation_call_id: co_active_info}
        
        # Build default phase schedule (maintains current behavior)
        self._build_default_phase_schedule()
        
    # For each set of layer chunks, take the index of the layer and the layer chunks and store them in the layer_chunks dict
    def set_layer_chunks(self, layer_idx, layer_chunks):
        # With a dictionary, we can directly assign to any key without range concerns
        self.layer_chunks[layer_idx] = layer_chunks
        
    # === NEW: Phase Schedule Building Methods ===
    def _build_default_phase_schedule(self):
        """Build traditional compute->activation_comm schedule"""
        for layer_idx in range(self.num_layers):
            # Add compute phase
            compute_phase = create_compute_phase(
                phase_id=self.next_phase_id,
                model_idx=self.model_idx,
                layer_idx=layer_idx,
                chiplet_assignments=None  # Will be set from mapping
            )
            compute_phase_id = self._add_phase_object(compute_phase)
            self.layer_to_compute_phase[layer_idx] = compute_phase_id
            
            # Add activation communication phase (except last layer)
            if layer_idx < self.num_layers - 1:
                comm_phase = create_activation_comm_phase(
                    phase_id=self.next_phase_id,
                    model_idx=self.model_idx,
                    layer_idx=layer_idx,
                    dependencies=[compute_phase_id]
                )
                comm_phase_id = self._add_phase_object(comm_phase)
                self.layer_to_activation_comm_phase[layer_idx] = comm_phase_id
        
        # Set dependencies for compute phases (each depends on previous layer's comm)
        for layer_idx in range(1, self.num_layers):
            compute_phase_id = self.layer_to_compute_phase[layer_idx]
            prev_comm_phase_id = self.layer_to_activation_comm_phase.get(layer_idx - 1)
            if prev_comm_phase_id is not None:
                self.phases[compute_phase_id].dependencies.append(prev_comm_phase_id)
        
        # Initialize phase instances for all inputs
        self._initialize_phase_instances()
    
    def _add_phase_object(self, phase: Phase) -> int:
        """Add a phase object and return its ID"""
        phase_id = phase.phase_id
        self.next_phase_id = max(self.next_phase_id, phase_id + 1)
        
        self.phases[phase_id] = phase
        self.phase_execution_order.append(phase_id)
        return phase_id
    
    def _initialize_phase_instances(self):
        """Initialize phase instances for all inputs"""
        for input_idx in range(self.num_inputs):
            for phase_id, phase in self.phases.items():
                self.phase_instances[input_idx][phase_id] = PhaseInstance(
                    phase_id=phase_id,
                    input_idx=input_idx,
                    state=PhaseState.NOT_STARTED,
                )
    
    def add_weight_loading_phases(self, weight_schedule: Dict[int, List[int]]):
        """Add weight loading phases to the schedule
        Args:
            weight_schedule: {before_layer_idx: [layer_indices_to_load]}
        """
        for before_layer, layers_to_load in weight_schedule.items():
            # Create weight loading phase (handles both single and multiple layers)
            weight_phase = create_weight_loading_phase(
                phase_id=self.next_phase_id,
                model_idx=self.model_idx,
                layers_to_load=layers_to_load
            )
            weight_phase_id = self._add_phase_object(weight_phase)
            
            # Map all layers to this phase
            for layer_idx in layers_to_load:
                self.layer_to_weight_comm_phase[layer_idx] = weight_phase_id
                
                # For JIT loading, compute phases should NOT depend on weight loading HERE
                # Weight loading dependencies will be handled separately in global_manager
                # This prevents circular dependencies and allows proper JIT coordination
        
        # Re-sort execution order to maintain dependencies
        self._sort_execution_order()
        
        # Re-initialize phase instances for new phases
        # When weight_stationary=True, weight loading phases should only exist for the first input
        # since subsequent inputs reuse the already-loaded weights
        for input_idx in range(self.num_inputs):
            for phase_id, phase in self.phases.items():
                if phase_id not in self.phase_instances[input_idx]:
                    # For weight loading phases in weight stationary mode, only create instances for input 0
                    if isinstance(phase, WeightLoadingPhase) and self.weight_stationary and input_idx > 0:
                        # Skip creating weight loading phase instances for subsequent inputs
                        # when weights are stationary (already loaded by first input)
                        continue
                    
                    self.phase_instances[input_idx][phase_id] = PhaseInstance(
                        phase_id=phase_id,
                        input_idx=input_idx,
                        state=PhaseState.NOT_STARTED,
                    )
    
    def _sort_execution_order(self):
        """Sort execution order based on dependencies (topological sort)"""
        # Simple topological sort
        sorted_order = []
        visited = set()
        
        def visit(phase_id):
            if phase_id in visited:
                return
            visited.add(phase_id)
            phase = self.phases[phase_id]
            for dep in phase.dependencies:
                visit(dep)
            sorted_order.append(phase_id)
        
        for phase_id in self.phases:
            visit(phase_id)
        
        self.phase_execution_order = sorted_order
        
    
    # === Traffic Calculation ===
    def calculate_all_traffic(self):
        """Calculate traffic for all communication phases using the traffic calculator"""
        for phase_id, phase in self.phases.items():
            if phase.can_generate_traffic():
                if isinstance(phase, WeightLoadingPhase):
                    if phase.is_combined():
                        print(f"  Phase {phase_id}: type={phase.get_phase_type_name()}, combined layers={phase.layers_to_load}")
                        # Calculate combined traffic for all layers
                        phase.traffic = self.traffic_calculator.calculate_weight_loading_traffic(
                            phase=phase,
                            model_metrics=self.model_metrics,
                            mapping=self.mapping
                        )
                    else:
                        # Single layer weight loading
                        layer_idx = phase.layers_to_load[0] if phase.layers_to_load else -1
                        print(f"  Phase {phase_id}: type={phase.get_phase_type_name()}, layer={layer_idx}")
                        phase.traffic = self.traffic_calculator.calculate_weight_loading_traffic(
                            phase=phase,
                            model_metrics=self.model_metrics,
                            mapping=self.mapping
                        )
                elif isinstance(phase, ActivationCommPhase):
                    print(f"  Phase {phase_id}: type={phase.get_phase_type_name()}, layer={phase.layer_idx}")
                    phase.traffic = self.traffic_calculator.calculate_activation_traffic(
                        phase=phase,
                        model_def=self.model_def,
                        model_metrics=self.model_metrics,
                        mapping=self.mapping
                    )
                
                if phase.traffic:
                    total_packets = sum(sum(dests.values()) for dests in phase.traffic.values())
                    print(f"    Traffic calculated: {len(phase.traffic)} sources, {total_packets} total packets")
                else:
                    print(f"    No traffic calculated for this phase")

    def add_chunk_compute_result(self, layer_idx, chiplet_id, chunk_idx, chunk_result):
        """
        Store simulation results for a specific layer chunk.
        
        Args:
            layer_idx (int): Index of the layer
            chiplet_id (int): ID of the chiplet that processed this chunk
            chunk_idx (int): Index of the chunk within the layer (for ordering)
            chunk_result (dict): Simulation results for this chunk (latency, energy, etc.)
        """
        # Create the layer entry in the results dictionary if it doesn't exist
        if layer_idx not in self.chunk_results:
            self.chunk_results[layer_idx] = {}
        
        # Store the result for this specific chiplet with chunk index
        # Use a tuple of (chunk_idx, result) to maintain ordering information
        self.chunk_results[layer_idx][chiplet_id] = (chunk_idx, chunk_result)
        
        # Update the chiplets used for this layer
        if layer_idx not in self.layer_chiplets:
            self.layer_chiplets[layer_idx] = set()
        self.layer_chiplets[layer_idx].add(chiplet_id)
    
    def update_all_layer_compute_metrics(self):
        """
        Updates layer compute latency and energy for all inputs based on chunk results.
        Assumes chunk compute results are identical across all inputs.
        Updates both legacy structures and new phase system.
        """
        # Process all layers that have chunk results
        for layer_idx in self.chunk_results.keys():
            # Calculate latency (max across all chunks for this layer)
            layer_compute_latency = 0
            for chiplet_id, (chunk_idx, result) in self.chunk_results[layer_idx].items():
                if 'latency_us' in result:
                    layer_compute_latency = max(layer_compute_latency, result['latency_us'])
            
            # Calculate energy (sum across all chunks for this layer)
            layer_compute_energy = 0
            for chiplet_id, (chunk_idx, result) in self.chunk_results[layer_idx].items():
                if 'energy_fj' in result:
                    layer_compute_energy += result['energy_fj']
            
            # Update phase instances with compute metrics
            if layer_idx in self.layer_to_compute_phase:
                compute_phase_id = self.layer_to_compute_phase[layer_idx]
                for input_idx in range(self.num_inputs):
                    instance = self.phase_instances[input_idx][compute_phase_id]
                    instance.latency_us = layer_compute_latency
                    instance.energy_fj = layer_compute_energy
    def get_chiplets_used(self):
        """
        Get a set of all chiplet IDs used across all layers.
        
        Returns:
            set: Set of all chiplet IDs used
        """
        all_chiplets = set()
        for chiplets in self.layer_chiplets.values():
            all_chiplets.update(chiplets)
        return all_chiplets
    
    def get_ordered_chunks(self, layer_idx):
        """
        Get the chunk results for a layer in order of their chunk indices.
        
        Args:
            layer_idx (int): Index of the layer
            
        Returns:
            list: List of (chiplet_id, chunk_result) tuples ordered by chunk_idx
        """
        if layer_idx not in self.chunk_results:
            return []
        
        # Extract all chiplet_id, (chunk_idx, result) pairs
        chunk_pairs = [(chiplet_id, chunk_data) for chiplet_id, chunk_data in self.chunk_results[layer_idx].items()]
        
        # Sort by chunk_idx (which is the first element of the second item in each pair)
        sorted_chunks = sorted(chunk_pairs, key=lambda x: x[1][0])
        
        # Return list of (chiplet_id, result) tuples (discarding the chunk_idx in the output)
        return [(chiplet_id, chunk_data[1]) for chiplet_id, chunk_data in sorted_chunks]
        
    # === NEW: Phase-based Methods ===
    def get_active_phases(self, global_time_us: float) -> List[PhaseInstance]:
        """
        Get all currently active phases with scaled traffic for partial completion.
        This maintains the critical functionality of scaling traffic based on completion.
        """
        active_phases = []
        
        for input_idx in range(self.num_inputs):
            # Skip if input hasn't started
            if self.input_start_time_us[input_idx] == -1:
                continue
            if global_time_us < self.input_start_time_us[input_idx]:
                continue
            
            # Find the current phase for this input
            current_phase_id = self._find_current_phase(input_idx, global_time_us)
            if current_phase_id is None:
                continue
                
            phase = self.phases[current_phase_id]
            instance = self.phase_instances[input_idx][current_phase_id]
            
            # Handle based on phase type
            if isinstance(phase, ComputePhase):
                # Compute phase handling
                if instance.state == PhaseState.NOT_STARTED:
                    instance.state = PhaseState.RUNNING
                    instance.start_time_us = self._calculate_phase_start_time(input_idx, current_phase_id)
                    instance.is_active = True
                    active_phases.append(instance)
                    
                elif instance.state == PhaseState.RUNNING:
                    # Check if still running
                    elapsed = global_time_us - instance.start_time_us
                    if elapsed < instance.latency_us:
                        instance.percent_complete = elapsed / instance.latency_us if instance.latency_us > 0 else 0
                        instance.is_active = True
                        active_phases.append(instance)
                    else:
                        instance.state = PhaseState.COMPLETE
                        instance.is_active = False
                        instance.completion_time_us = instance.start_time_us + instance.latency_us
                        
            elif phase.can_generate_traffic():
                # Communication phase handling with traffic scaling
                
                # Special handling for weight loading in stationary mode
                if isinstance(phase, WeightLoadingPhase):
                    # Get layers to load from the weight loading phase
                    layers_to_load = phase.layers_to_load
                    
                    
                    # Check if ALL weights for this phase are already loaded
                    all_weights_loaded = self.weight_stationary and all(
                        self.weights_loaded.get(layer_idx, False) for layer_idx in layers_to_load
                    )
                    
                    if all_weights_loaded:
                        # All weights already loaded - skip weight loading
                        if instance.state == PhaseState.NOT_STARTED:
                            # Mark as instantly complete with zero latency
                            instance.state = PhaseState.COMPLETE
                            instance.latency_us = 0
                            instance.start_time_us = global_time_us
                            instance.completion_time_us = global_time_us
                            instance.is_active = False
                        print(f"  Skipping - all weights already loaded")
                        continue  # Don't add to active phases
                
                # Case 1: New communication starting (latency == 0)
                if instance.latency_us == 0 and instance.state == PhaseState.NOT_STARTED:
                    # Get the traffic for this phase
                    instance.scaled_traffic = self._get_phase_traffic(phase, 1.0)
                    
                    # Check if there's actual traffic to simulate
                    if not self._has_nonzero_traffic(instance.scaled_traffic):
                        # No actual traffic - mark as no communication (-1 latency)
                        instance.state = PhaseState.COMPLETE
                        instance.start_time_us = self._calculate_phase_start_time(input_idx, current_phase_id)
                        instance.latency_us = -1  # Use -1 to indicate no communication
                        instance.completion_time_us = instance.start_time_us
                        instance.is_active = False
                        instance.percent_complete = 1.0
                        
                        # Mark weights as loaded for weight loading phases
                        if isinstance(phase, WeightLoadingPhase):
                            for layer_idx in phase.layers_to_load:
                                self.weights_loaded[layer_idx] = True
                                print(f"  Marked weights as loaded for layer {layer_idx}")
                    else:
                        # Has traffic - proceed normally
                        instance.state = PhaseState.RUNNING
                        instance.start_time_us = self._calculate_phase_start_time(input_idx, current_phase_id)
                        instance.is_active = True
                        instance.percent_complete = 0.0
                        active_phases.append(instance)
                    
                # Case 2: Ongoing communication
                elif instance.state == PhaseState.RUNNING:
                    # Check if phase has received a latency update
                    if instance.latency_us > 0:
                        # Phase has a latency - check if it's complete
                        elapsed = global_time_us - instance.start_time_us
                        
                        if elapsed < instance.latency_us:
                            # Still active - calculate scaled traffic
                            instance.percent_complete = min(1.0, elapsed / instance.latency_us)
                            remaining_fraction = 1.0 - instance.percent_complete
                            
                            # Scale traffic based on remaining work
                            instance.scaled_traffic = self._get_phase_traffic(phase, remaining_fraction)
                            
                            # Only add if there's actual traffic
                            if self._has_nonzero_traffic(instance.scaled_traffic):
                                active_phases.append(instance)
                        else:
                            # Communication complete
                            instance.state = PhaseState.COMPLETE
                            instance.is_active = False
                            instance.completion_time_us = instance.start_time_us + instance.latency_us
                            
                            # Mark weights as loaded for weight loading phases
                            if isinstance(phase, WeightLoadingPhase):
                                for layer_idx in phase.layers_to_load:
                                    self.weights_loaded[layer_idx] = True
                    else:
                        # Phase is running but hasn't received latency update yet
                        # Check if this phase has traffic to simulate
                        instance.scaled_traffic = self._get_phase_traffic(phase, 1.0)
                        
                        if not self._has_nonzero_traffic(instance.scaled_traffic):
                            # No traffic - complete immediately
                            instance.state = PhaseState.COMPLETE
                            instance.is_active = False
                            instance.completion_time_us = instance.start_time_us
                            instance.latency_us = -1  # Use -1 to indicate no communication
                            instance.percent_complete = 1.0
                            
                            # Mark weights as loaded for weight loading phases
                            if isinstance(phase, WeightLoadingPhase):
                                for layer_idx in phase.layers_to_load:
                                    self.weights_loaded[layer_idx] = True
                        else:
                            # Has traffic - add to active phases for simulation
                            active_phases.append(instance)
        
        return active_phases
        
    def _find_current_phase(self, input_idx: int, global_time_us: float) -> Optional[int]:
        """Find which phase should be active for this input at this time"""
        for phase_id in self.phase_execution_order:
            # Skip if phase instance doesn't exist for this input
            if phase_id not in self.phase_instances[input_idx]:
                continue
                
            instance = self.phase_instances[input_idx][phase_id]
            
            # If phase is not complete, check if it can run
            if instance.state != PhaseState.COMPLETE:
                # Check dependencies
                if self._dependencies_met(input_idx, phase_id):
                    return phase_id
        return None
    
    def _dependencies_met(self, input_idx: int, phase_id: int) -> bool:
        """Check if all dependencies for a phase are met"""
        phase = self.phases[phase_id]
        
        # Check normal dependencies first
        for dep_id in phase.dependencies:
            # Skip if dependency phase instance doesn't exist for this input
            if dep_id not in self.phase_instances[input_idx]:
                continue
            dep_instance = self.phase_instances[input_idx][dep_id]
            if dep_instance.state != PhaseState.COMPLETE:
                return False
        
        # For pipelined execution, also check inter-input dependencies
        if self.communication_method == "pipelined" and input_idx > 0:
            # This input's phase can't start until previous input completes same phase
            # Skip if phase instance doesn't exist for previous input
            if phase_id not in self.phase_instances[input_idx - 1]:
                # If the phase doesn't exist for the previous input, it means it was skipped
                # (e.g., weight loading phases in weight stationary mode)
                # In this case, we can proceed
                pass
            else:
                prev_instance = self.phase_instances[input_idx - 1][phase_id]
                if prev_instance.state != PhaseState.COMPLETE:
                    return False
        
        # Special validation for COMPUTE phases - verify weights are actually loaded
        # This check happens AFTER dependency checks, so if we reach here and weights
        # aren't loaded, it means there's a configuration error
        if isinstance(phase, ComputePhase):
            layer_idx = phase.layer_idx
            
            # Check if there's a weight loading phase for this layer
            weight_phase_id = self.layer_to_weight_comm_phase.get(layer_idx)
            if weight_phase_id is not None:
                # Weight loading phase exists - check if it's complete or weights are loaded
                weight_instance = self.phase_instances[input_idx].get(weight_phase_id)
                
                # In weight stationary mode, weights might already be loaded from a previous input
                if not (self.weight_stationary and self.weights_loaded[layer_idx]):
                    # Weights not already loaded - check if weight loading phase is complete
                    # If dependencies are met but weight phase isn't complete, there's a problem
                    if weight_instance is None or weight_instance.state != PhaseState.COMPLETE:
                        # This should not happen if dependencies are set up correctly
                        # The weight loading phase should be in the dependencies
                        if weight_phase_id not in phase.dependencies:
                            raise RuntimeError(
                                f"CONFIGURATION ERROR: Compute phase for layer {layer_idx} "
                                f"(phase {phase_id}) does not depend on weight loading phase {weight_phase_id}!\n"
                                f"  Model: {self.model_name} (idx: {self.model_idx})\n"
                                f"  This is a bug in the weight loading schedule setup."
                            )
                else:
                    # In weight stationary mode with weights already loaded, we need to check
                    # if the weight loading phase instance exists for this input
                    if weight_instance is None:
                        # Weight loading phase instance doesn't exist for this input in weight stationary mode
                        # This is expected for subsequent inputs - they don't need to load weights again
                        # The weights are already loaded by the first input
                        pass  # This is the correct behavior
        
        return True
    
    def _calculate_phase_start_time(self, input_idx: int, phase_id: int) -> float:
        """Calculate when a phase can start based on dependencies"""
        phase = self.phases[phase_id]
        start_time = self.input_start_time_us[input_idx]
        
        # Consider dependency completion times
        for dep_id in phase.dependencies:
            dep_instance = self.phase_instances[input_idx].get(dep_id)
            if dep_instance is not None and dep_instance.completion_time_us > start_time:
                start_time = dep_instance.completion_time_us
        
        # For pipelined execution
        if self.communication_method == "pipelined" and input_idx > 0:
            prev_instance = self.phase_instances[input_idx - 1].get(phase_id)
            if prev_instance is not None and prev_instance.completion_time_us > start_time:
                start_time = prev_instance.completion_time_us
        
        return start_time
    
    def _get_phase_traffic(self, phase: Phase, scale_factor: float) -> TrafficMatrixDict:
        """Get traffic for a phase, optionally scaled"""
        if phase.traffic is None:
            return {}
            
        if scale_factor >= 1.0:
            return phase.traffic.copy()
            
        # Scale traffic
        scaled = {}
        for src, dests in phase.traffic.items():
            scaled[src] = {}
            for dst, amount in dests.items():
                if amount > 0:
                    # Use rounding with minimum of 1 packet to avoid losing traffic
                    scaled_amount = max(1, round(amount * scale_factor))
                    scaled[src][dst] = scaled_amount
        return scaled
    
    def _has_nonzero_traffic(self, traffic: TrafficMatrixDict) -> bool:
        """Check if traffic matrix has any non-zero values"""
        for src, dests in traffic.items():
            for dst, amount in dests.items():
                if amount > 0:
                    return True
        return False
        
    def update_phase_latency(self, input_idx: int, phase_id: int, new_latency_us: float, 
                             global_time_us: float):
        """
        Update communication latency dynamically based on network conditions.
        This maintains the critical functionality of updating latencies multiple times.
        """
        # Skip if phase instance doesn't exist for this input
        if phase_id not in self.phase_instances[input_idx]:
            return
            
        instance = self.phase_instances[input_idx][phase_id]
        old_latency = instance.latency_us
        
        # Store history
        instance.latency_history.append((global_time_us, new_latency_us))
        
        # Setup debug logger (once)
        logger = logging.getLogger('phase_debug')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logs_dir = os.path.join(os.getcwd(), 'temp', 'logs')
            try:
                os.makedirs(logs_dir, exist_ok=True)
            except Exception:
                pass
            fh = logging.FileHandler(os.path.join(logs_dir, 'phase_debug.log'))
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s | %(message)s')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        # Gather context for logging
        phase = self.phases.get(phase_id)
        phase_type = phase.get_phase_type_name() if phase else "Unknown"
        layer_idx = getattr(phase, 'layer_idx', -1) if phase else -1
        # Best-effort packet total used for this phase (from last scaled_traffic if present)
        packet_total = 0
        if hasattr(instance, 'scaled_traffic') and instance.scaled_traffic:
            for dests in instance.scaled_traffic.values():
                if dests:
                    packet_total += sum(max(0, v) for v in dests.values())
        elif phase and hasattr(phase, 'traffic') and phase.traffic:
            for dests in phase.traffic.values():
                if dests:
                    packet_total += sum(max(0, v) for v in dests.values())

        # Update latency
        if old_latency == 0:
            # First simulation result
            instance.latency_us = new_latency_us
            logger.info(
                f"initial_update | model_idx={self.model_idx} input_idx={input_idx} phase_id={phase_id} "
                f"phase_type={phase_type} layer_idx={layer_idx} packets={packet_total} "
                f"old_latency_us={old_latency:.3f} new_latency_us={new_latency_us:.3f} elapsed_us=0.000"
            )
            
        elif old_latency > 0 and instance.state == PhaseState.RUNNING:
            # Update for ongoing communication
            elapsed = global_time_us - instance.start_time_us
            # New total = elapsed + new simulation of remaining scaled traffic
            new_total = elapsed + new_latency_us
            
            instance.latency_us = new_total
            
            # Recalculate percent_complete after latency update to maintain consistency
            if instance.latency_us > 0:
                instance.percent_complete = min(1.0, elapsed / instance.latency_us)
            logger.info(
                f"ongoing_update | model_idx={self.model_idx} input_idx={input_idx} phase_id={phase_id} "
                f"phase_type={phase_type} layer_idx={layer_idx} packets={packet_total} "
                f"old_latency_us={old_latency:.3f} new_simulated_us={new_latency_us:.3f} elapsed_us={elapsed:.3f} "
                f"new_total_us={new_total:.3f}"
            )
    
    def calculate_phase_timing(self, global_time_us: float) -> bool:
        """
        Calculate start and completion times for all phases.
        Returns True if any timing was updated.
        """
        updated = False

        for input_idx in range(self.num_inputs):
            # Handle input start time (for pipelined execution)
            if input_idx > 0 and self.input_start_time_us[input_idx] == -1:
                if self.communication_method == "pipelined":
                    # Wait for first layer completion of previous input (compute + activation comm phases)
                    
                    # Find the first layer that has both compute and activation comm phases in the previous input
                    first_layer_start_time = None
                    for layer_idx in sorted(self.layer_to_compute_phase.keys()):
                        compute_phase_id = self.layer_to_compute_phase[layer_idx]
                        activation_comm_phase_id = self.layer_to_activation_comm_phase.get(layer_idx)
                        
                        # Check if both phases exist in the previous input
                        if (compute_phase_id in self.phase_instances[input_idx - 1] and 
                            (activation_comm_phase_id is None or activation_comm_phase_id in self.phase_instances[input_idx - 1])):
                            
                            # Get the compute phase instance
                            compute_instance = self.phase_instances[input_idx - 1][compute_phase_id]
                            
                            # Determine layer completion time
                            if activation_comm_phase_id is not None:
                                # Layer has activation communication - wait for both to complete
                                activation_instance = self.phase_instances[input_idx - 1][activation_comm_phase_id]
                                if (compute_instance.state == PhaseState.COMPLETE and 
                                    activation_instance.state == PhaseState.COMPLETE):
                                    layer_completion_time = max(compute_instance.completion_time_us, 
                                                              activation_instance.completion_time_us)
                                    first_layer_start_time = layer_completion_time
                                    break
                            else:
                                # Layer has no activation communication - just wait for compute
                                if compute_instance.state == PhaseState.COMPLETE:
                                    first_layer_start_time = compute_instance.completion_time_us
                                    break
                    
                    if first_layer_start_time is not None:
                        self.input_start_time_us[input_idx] = first_layer_start_time
                        updated = True
                else:
                    # Non-pipelined: wait for previous input's phases to complete
                    # Check if all phases for the previous input are complete
                    prev_input_idx = input_idx - 1
                    if prev_input_idx in self.phase_instances:
                        existing_phases = [pid for pid in self.phase_execution_order if pid in self.phase_instances[prev_input_idx]]
                        all_prev_phases_complete = all(
                            self.phase_instances[prev_input_idx][pid].state == PhaseState.COMPLETE
                            for pid in existing_phases
                        )
                        if all_prev_phases_complete and self.input_start_time_us[input_idx] == -1:
                            # Start this input when previous input's phases are complete
                            if self.input_completion_time_us[prev_input_idx] != -1:
                                self.input_start_time_us[input_idx] = self.input_completion_time_us[prev_input_idx]
                                updated = True
            
            # Skip if input hasn't started
            if self.input_start_time_us[input_idx] == -1:
                continue
            
            # Process phases in execution order
            for phase_id in self.phase_execution_order:
                # Skip if phase instance doesn't exist for this input
                if phase_id not in self.phase_instances[input_idx]:
                    continue
                    
                instance = self.phase_instances[input_idx][phase_id]
                
                # Skip if already complete
                if instance.state == PhaseState.COMPLETE:
                    continue

                # Check dependencies
                if not self._dependencies_met(input_idx, phase_id):
                    continue

                # Calculate start time if not set
                if instance.start_time_us < 0:
                    instance.start_time_us = self._calculate_phase_start_time(input_idx, phase_id)
                    updated = True
                    
                # Calculate completion time if latency is known
                if instance.latency_us > 0 and instance.completion_time_us < 0:
                    instance.completion_time_us = instance.start_time_us + instance.latency_us
                    updated = True
                
            # Check if all phases complete for this input
            # Only consider phases that exist for this input
            existing_phases = [pid for pid in self.phase_execution_order if pid in self.phase_instances[input_idx]]
            all_complete = all(
                self.phase_instances[input_idx][pid].state == PhaseState.COMPLETE
                for pid in existing_phases
            )
            if all_complete and self.input_completion_time_us[input_idx] == -1:
                # Find the last completion time
                last_time = max(
                    self.phase_instances[input_idx][pid].completion_time_us
                    for pid in existing_phases
                )
                self.input_completion_time_us[input_idx] = last_time
                updated = True
        
        # Update model completion time
        if all(t != -1 for t in self.input_completion_time_us):
            new_completion = max(self.input_completion_time_us)
            if self.model_completion_time_us != new_completion:
                self.model_completion_time_us = new_completion
                updated = True

        return updated
        
    def mark_layers_without_communication(self):
        """
        Mark layers that have no communication traffic with -1 latency.
        This is important for post-processing tools that expect -1 to indicate
        no communication (as opposed to 0 latency which could mean instant communication).
        
        This affects:
        - The last layer (which has no activation communication to subsequent layers)
        - Any layers that have no receiving layers in the model definition
        - Layers that have activation communication phases but zero traffic (e.g., mapped to same chiplet)
        """
        for input_idx in range(self.num_inputs):
            for layer_idx in range(self.num_layers):
                # Check if this layer has an activation communication phase
                if layer_idx not in self.layer_to_activation_comm_phase:
                    # No activation communication phase for this layer
                    # Mark with -1 to indicate no communication
                    self.layer_activation_comm_latency[input_idx][layer_idx] = -1
                    self.layer_activation_comm_latency_history[input_idx][layer_idx] = []
                    self.layer_activation_comm_is_active[input_idx][layer_idx] = False
                    
                    # Also ensure activation communication traffic is empty for this layer
                    if layer_idx not in self.layer_activation_comm_traffic:
                        self.layer_activation_comm_traffic[layer_idx] = {}
                else:
                    # Layer has an activation communication phase - check if it has zero traffic
                    comm_phase_id = self.layer_to_activation_comm_phase[layer_idx]
                    comm_phase = self.phases[comm_phase_id]
                    
                    # Check if the phase has zero traffic
                    if comm_phase.traffic is None or not self._has_nonzero_traffic(comm_phase.traffic):
                        # Phase exists but has no traffic - mark with -1 to indicate no communication
                        self.layer_activation_comm_latency[input_idx][layer_idx] = -1
                        self.layer_activation_comm_latency_history[input_idx][layer_idx] = []
                        self.layer_activation_comm_is_active[input_idx][layer_idx] = False
                        
                        # Ensure activation communication traffic is empty for this layer
                        if layer_idx not in self.layer_activation_comm_traffic:
                            self.layer_activation_comm_traffic[layer_idx] = {}
    
    def sync_phase_metrics_to_layers(self):
        """
        Synchronize phase-level metrics to layer-level metrics.
        This aggregates metrics from all phases associated with each layer.
        """
        for input_idx in range(self.num_inputs):
            for phase_id, phase in self.phases.items():
                # Skip if phase instance doesn't exist for this input
                # This can happen for weight loading phases in weight stationary mode
                if phase_id not in self.phase_instances[input_idx]:
                    continue
                    
                instance = self.phase_instances[input_idx][phase_id]
                
                # Handle different phase types
                if isinstance(phase, ComputePhase):
                    layer_idx = phase.layer_idx
                    self.layer_compute_latency[input_idx][layer_idx] = instance.latency_us
                    self.layer_compute_energy[input_idx][layer_idx] = instance.energy_fj
                    
                elif isinstance(phase, ActivationCommPhase):
                    layer_idx = phase.layer_idx
                    self.layer_activation_comm_latency[input_idx][layer_idx] = instance.latency_us
                    
                    # Update latency history
                    if layer_idx not in self.layer_activation_comm_latency_history[input_idx]:
                        self.layer_activation_comm_latency_history[input_idx][layer_idx] = []
                    self.layer_activation_comm_latency_history[input_idx][layer_idx] = instance.latency_history.copy()
                    
                    # Update active status
                    self.layer_activation_comm_is_active[input_idx][layer_idx] = instance.is_active
                    
                elif isinstance(phase, WeightLoadingPhase):
                    # For weight loading, update metrics for all layers being loaded
                    for layer_idx in phase.layers_to_load:
                        self.layer_weight_loading_latency[input_idx][layer_idx] = instance.latency_us
                        
                        # Update latency history
                        if layer_idx not in self.layer_weight_loading_latency_history[input_idx]:
                            self.layer_weight_loading_latency_history[input_idx][layer_idx] = []
                        self.layer_weight_loading_latency_history[input_idx][layer_idx] = instance.latency_history.copy()
                        
                        # Update active status
                        self.layer_weight_loading_is_active[input_idx][layer_idx] = instance.is_active
                else:
                    continue  # Skip unknown phase types
                
                # Update layer timing for phases that have layer associations
                # Get layer indices for this phase
                layer_indices = []
                if isinstance(phase, (ComputePhase, ActivationCommPhase)):
                    layer_indices = [phase.layer_idx]
                elif isinstance(phase, WeightLoadingPhase):
                    layer_indices = phase.layers_to_load
                
                # Update timing for all associated layers
                for layer_idx in layer_indices:
                    if instance.start_time_us >= 0:
                        if layer_idx not in self.layer_start_time_us[input_idx] or \
                           self.layer_start_time_us[input_idx][layer_idx] < 0 or \
                           instance.start_time_us < self.layer_start_time_us[input_idx][layer_idx]:
                            self.layer_start_time_us[input_idx][layer_idx] = instance.start_time_us
                    
                    if instance.completion_time_us >= 0:
                        if layer_idx not in self.layer_completion_time_us[input_idx] or \
                           instance.completion_time_us > self.layer_completion_time_us[input_idx][layer_idx]:
                            self.layer_completion_time_us[input_idx][layer_idx] = instance.completion_time_us
        
        # Sync traffic (same for all inputs, so just use phase traffic directly)
        for phase_id, phase in self.phases.items():
            if isinstance(phase, ActivationCommPhase) and phase.traffic:
                # For activation communication, store in activation communication traffic
                self.layer_activation_comm_traffic[phase.layer_idx] = phase.traffic.copy()
            elif isinstance(phase, WeightLoadingPhase) and phase.traffic:
                # For weight loading, store traffic for all layers being loaded
                for layer_idx in phase.layers_to_load:
                    self.layer_weight_loading_traffic[layer_idx] = phase.traffic.copy()

    def is_complete(self, global_time_us, input_idx=None):
        """
        Check if the model (or a specific input) is complete based on the current simulation time,
        by comparing against pre-calculated completion times stored in attributes.
        
        Important: Ensure `calculate_completion_times()` has been called successfully *before* relying on this method,
                   otherwise completion time attributes might be -1.
        
        Args:
            global_time_us (float): Current global simulation time in microseconds.
            input_idx (int, optional): If provided, checks if this specific input is complete. 
                                       If None, checks if the entire model (all inputs) is complete. Defaults to None.
                                       
        Returns:
            bool: True if the specified scope (input or model) is complete according to its stored completion time, 
                  False otherwise (including if completion time is -1, meaning not yet calculated or calculation failed).
        """
        if input_idx is not None:
            # Check completion for a specific input by accessing the attribute
            if not (0 <= input_idx < self.num_inputs):
                 raise IndexError(f"Invalid input_idx {input_idx} provided for model '{self.model_name}'. Must be between 0 and {self.num_inputs - 1}.")
                 
            comp_time = self.input_completion_time_us[input_idx]
            if comp_time == -1:
                # Not yet calculated or failed calculation, so cannot be complete
                return False 
            return global_time_us >= comp_time
        else:
            # Check completion for the entire model by accessing the attribute
            comp_time = self.model_completion_time_us
            if comp_time == -1:
                 # Not yet calculated or failed calculation, so cannot be complete
                return False
            return global_time_us >= comp_time
    
    def record_co_active_traffic(self, simulation_call_id: int, simulation_time_us: float, 
                                co_active_phases: List[Dict]):
        """
        Record co-active traffic information for a simulation call.
        
        Args:
            simulation_call_id (int): Unique identifier for this simulation call
            simulation_time_us (float): Global time when simulation was called
            co_active_phases (List[Dict]): List of phase information for all phases simulated together
                Each dict should contain:
                - model_idx: int
                - input_idx: int  
                - phase_id: int
                - phase_type: str ('ACTIVATION_COMM' or 'WEIGHT_LOADING_COMM')
                - layer_idx: int (for activation comm phases)
                - next_compute_layer: int (for weight loading phases)
                - traffic_contribution: Dict[src_chiplet, Dict[dst_chiplet, bytes]]
        """
        self.communication_co_active_traffic[simulation_call_id] = {
            'simulation_time_us': simulation_time_us,
            'phases_simulated_together': co_active_phases.copy()
        }
        
        