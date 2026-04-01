# temporal_filter.py

from typing import Dict, Callable


class TemporalFilter:
    """
    Filters simulation data based on temporal criteria (warmup and cooldown periods).
    
    Responsibilities:
    - Filter models by warmup period (exclude early models)
    - Filter models by cooldown period (exclude late models)
    
    Args:
        print_header_func: Function to print formatted headers (optional)
    """
    
    def __init__(self, print_header_func=None):
        """
        Initialize the temporal filter.
        
        Args:
            print_header_func: Optional function for printing formatted headers
        """
        self._print_header = print_header_func or self._default_print_header
    
    def _default_print_header(self, title, char="═", box_width=53):
        """Default header printer if none provided"""
        print("\n" + "=" * 80)
        print(f"╔{char * (box_width)}╗")
        print(f"║{title.center(box_width)}║")
        print(f"╚{char * (box_width)}╝")
    
    def _apply_filter(
        self,
        retired_models: Dict,
        filter_predicate: Callable,
        filter_name: str,
        description: str,
        no_filter_message: str,
        warning_message: str
    ) -> Dict:
        """
        Generic method to apply a temporal filter to retired models.
        
        Args:
            retired_models: Dictionary of retired mapped models
            filter_predicate: Function that takes a mapped_model and returns True if it should be kept
            filter_name: Name of the filter (for messages)
            description: Description of what's being kept
            no_filter_message: Message to print if filter is not active
            warning_message: Warning message if all models are filtered out
            
        Returns:
            Dictionary of filtered models
        """
        filtered_models = {}
        original_count = len(retired_models)
        
        # Apply the filter predicate to each model
        for model_idx, mapped_model in retired_models.items():
            if filter_predicate(mapped_model):
                filtered_models[model_idx] = mapped_model
        
        filtered_count = len(filtered_models)
        removed_count = original_count - filtered_count
        
        # Print results
        if removed_count > 0:
            print(f"{filter_name} applied: Kept {filtered_count} models, removed {removed_count} models {description}.")
        else:
            print(f"All {original_count} retired models are included for subsequent analysis.")
        
        # Handle edge cases
        if filtered_count == 0 and original_count > 0:
            print(f"⚠️ WARNING: {warning_message}")
            exit(1)
        elif filtered_count < original_count:
            print(f"Proceeding with {filtered_count} models for subsequent analysis.")
        
        return filtered_models
    
    def filter_by_warmup(
        self, 
        retired_models: Dict, 
        warmup_period_us: float
    ) -> Dict:
        """
        Filter retired models, keeping only those that started at or after the warmup period.
        
        Args:
            retired_models: Dictionary of retired mapped models
            warmup_period_us: Warmup period threshold in microseconds
            
        Returns:
            Dictionary of filtered models that started after warmup period
        """
        self._print_header("APPLYING WARMUP PERIOD FILTER")
        
        if warmup_period_us <= 0:
            print("Warmup period is 0 μs. No filtering applied.")
            return retired_models.copy()
        
        print(f"Filtering retired models. Keeping models started at or after {warmup_period_us:.2f} μs.")
        
        return self._apply_filter(
            retired_models=retired_models,
            filter_predicate=lambda model: model.model_start_time_us >= warmup_period_us,
            filter_name="Warmup filter",
            description="started before the warmup period",
            no_filter_message="Warmup period is 0 μs. No filtering applied.",
            warning_message="All retired models were removed by the warmup filter. No models remain for post-warmup analysis."
        )
    
    def filter_by_cooldown(
        self,
        retired_models: Dict,
        cooldown_period_us: float,
        simulation_end_time_us: float
    ) -> Dict:
        """
        Filter retired models, keeping only those that completed before the cooldown period.
        
        The cooldown period is measured from the end of the simulation backwards.
        Models that completed during the cooldown period are excluded.
        
        Args:
            retired_models: Dictionary of retired mapped models
            cooldown_period_us: Cooldown period threshold in microseconds
            simulation_end_time_us: Total simulation time in microseconds
            
        Returns:
            Dictionary of filtered models that completed before cooldown period
        """
        self._print_header("APPLYING COOLDOWN PERIOD FILTER")
        
        if cooldown_period_us <= 0:
            print("Cooldown period is 0 μs. No filtering applied.")
            return retired_models.copy()
        
        cooldown_start_time = simulation_end_time_us - cooldown_period_us
        print(f"Filtering retired models. Keeping models completed before {cooldown_start_time:.2f} μs.")
        print(f"(Excluding models that completed in the last {cooldown_period_us:.2f} μs of simulation)")
        
        return self._apply_filter(
            retired_models=retired_models,
            filter_predicate=lambda model: model.completion_time_us <= cooldown_start_time,
            filter_name="Cooldown filter",
            description="completed during cooldown period",
            no_filter_message="Cooldown period is 0 μs. No filtering applied.",
            warning_message="All retired models were removed by the cooldown filter. No models remain for analysis."
        )
    
    def filter_by_warmup_and_cooldown(
        self,
        retired_models: Dict,
        warmup_period_us: float,
        cooldown_period_us: float,
        simulation_end_time_us: float
    ) -> Dict:
        """
        Apply both warmup and cooldown filters sequentially.
        
        Args:
            retired_models: Dictionary of retired mapped models
            warmup_period_us: Warmup period threshold in microseconds
            cooldown_period_us: Cooldown period threshold in microseconds
            simulation_end_time_us: Total simulation time in microseconds
            
        Returns:
            Dictionary of filtered models that meet both criteria
        """
        # First apply warmup filter
        models_after_warmup = self.filter_by_warmup(retired_models, warmup_period_us)
        
        # Then apply cooldown filter
        models_after_both = self.filter_by_cooldown(
            models_after_warmup, 
            cooldown_period_us, 
            simulation_end_time_us
        )
        
        return models_after_both
