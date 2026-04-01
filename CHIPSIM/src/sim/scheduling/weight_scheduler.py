"""
Weight Scheduler Module

Handles weight loading schedule generation and phase dependency configuration
for different loading strategies (all_at_once vs just_in_time).
"""

from typing import Dict, List


class WeightScheduler:
    """
    Generates weight loading schedules and configures phase dependencies.
    
    Supports two strategies:
    - 'all_at_once': Load all weights before first layer compute
    - 'just_in_time': Load each layer's weights just before its compute phase
    
    Args:
        strategy (str): Weight loading strategy ('all_at_once' or 'just_in_time')
    """
    
    def __init__(self, strategy: str = "all_at_once"):
        """
        Initialize the WeightScheduler.
        
        Args:
            strategy (str): Weight loading strategy
            
        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in ["all_at_once", "just_in_time"]:
            raise ValueError(
                f"Invalid weight_loading_strategy: {strategy}. "
                f"Must be 'all_at_once' or 'just_in_time'"
            )
        self.strategy = strategy
    
    def generate_schedule(self, num_layers: int) -> Dict[int, List[int]]:
        """
        Generate weight loading schedule based on the configured strategy.
        
        Args:
            num_layers (int): Number of layers in the model
            
        Returns:
            Dict[int, List[int]]: Weight loading schedule {before_layer_idx: [layer_indices_to_load]}
        """
        if self.strategy == "all_at_once":
            # Load all weights before the first layer's compute phase
            return {0: list(range(num_layers))}
            
        elif self.strategy == "just_in_time":
            # Load each layer's weights just before its compute phase
            schedule = {}
            for layer_idx in range(num_layers):
                schedule[layer_idx] = [layer_idx]
            return schedule
            
        else:
            raise ValueError(f"Unknown weight loading strategy: {self.strategy}")
    
    def configure_dependencies(self, mapped_model):
        """
        Configure dependencies for the selected weight loading strategy.
        
        Delegates to strategy-specific configuration methods.
        
        Args:
            mapped_model: MappedModel instance to configure
        """
        if self.strategy == "just_in_time":
            self._configure_jit_dependencies(mapped_model)
        elif self.strategy == "all_at_once":
            self._configure_all_at_once_dependencies(mapped_model)
    
    def _configure_all_at_once_dependencies(self, mapped_model):
        """
        Configure dependencies for all-at-once weight loading strategy.
        
        For all-at-once loading, all weights are loaded before the first layer computes.
        This method ensures:
        1. All compute phases depend on the single weight loading phase
        2. The weight loading phase has no dependencies (starts immediately)
        
        Args:
            mapped_model: MappedModel instance to configure
        """
        # Find the weight loading phase for layer 0 (which loads all weights)
        weight_phase_id = mapped_model.layer_to_weight_comm_phase.get(0)
        
        if weight_phase_id is None:
            print("⚠️ WARNING: No weight loading phase found for all-at-once strategy")
            return
        
        # Make all compute phases depend on this single weight loading phase
        for layer_idx in range(mapped_model.num_layers):
            compute_phase_id = mapped_model.layer_to_compute_phase.get(layer_idx)
            
            if compute_phase_id is not None:
                compute_phase = mapped_model.phases[compute_phase_id]
                if weight_phase_id not in compute_phase.dependencies:
                    compute_phase.dependencies.append(weight_phase_id)
        
        print(f"    ✅ All-at-once: All {mapped_model.num_layers} compute phases now depend on weight loading phase {weight_phase_id}")
        
        # Re-sort execution order after adding dependencies
        mapped_model._sort_execution_order()
    
    def _configure_jit_dependencies(self, mapped_model):
        """
        Configure dependencies for just-in-time weight loading strategy.
        
        For JIT loading, this method adds two critical dependencies:
        1. Each weight loading phase (except layer 0) depends on the previous layer's
           activation communication (if exists) or compute phase (if no activation)
        2. Each compute phase depends on its corresponding weight loading phase
        
        Args:
            mapped_model: MappedModel instance to configure
        """
        
        # Step 1: Each weight loading phase (except layer 0) depends on the previous layer's
        # activation communication or compute phase
        for layer_idx in range(1, mapped_model.num_layers):
            # Get the weight loading phase for this layer
            weight_phase_id = mapped_model.layer_to_weight_comm_phase.get(layer_idx)
            if weight_phase_id is None:
                continue
            
            weight_phase = mapped_model.phases[weight_phase_id]
            
            # Check if previous layer has activation communication
            prev_act_phase_id = mapped_model.layer_to_activation_comm_phase.get(layer_idx - 1)
            
            if prev_act_phase_id is not None:
                # Activation phase exists - depend on it
                if prev_act_phase_id not in weight_phase.dependencies:
                    weight_phase.dependencies.append(prev_act_phase_id)
            else:
                # No activation phase - depend on compute instead
                prev_compute_phase_id = mapped_model.layer_to_compute_phase.get(layer_idx - 1)
                if prev_compute_phase_id is not None:
                    if prev_compute_phase_id not in weight_phase.dependencies:
                        weight_phase.dependencies.append(prev_compute_phase_id)
        
        # Step 2: CRITICAL FIX - Each compute phase depends on its corresponding weight loading phase
        for layer_idx in range(mapped_model.num_layers):
            compute_phase_id = mapped_model.layer_to_compute_phase.get(layer_idx)
            weight_phase_id = mapped_model.layer_to_weight_comm_phase.get(layer_idx)
            
            if compute_phase_id is not None and weight_phase_id is not None:
                compute_phase = mapped_model.phases[compute_phase_id]
                if weight_phase_id not in compute_phase.dependencies:
                    compute_phase.dependencies.append(weight_phase_id)
        
        # Re-sort execution order after adding dependencies
        mapped_model._sort_execution_order()
    
    def print_schedule_summary(self, mapped_model, model_idx: int):
        """
        Print a summary of the phase execution order for debugging.
        
        Args:
            mapped_model: MappedModel instance
            model_idx: Model index for identification
        """
        print(f"    📋 Phase execution order for model {model_idx}:")
        print(f"    Phase IDs: {mapped_model.phase_execution_order}")
        
        for phase_id in mapped_model.phase_execution_order:
            phase = mapped_model.phases[phase_id]
            deps = [f"Phase {dep_id}" for dep_id in phase.dependencies]
            deps_str = f" (deps: {', '.join(deps)})" if deps else " (no dependencies)"
            print(f"      Phase {phase_id} ({phase.get_phase_type_name()}){deps_str}")

