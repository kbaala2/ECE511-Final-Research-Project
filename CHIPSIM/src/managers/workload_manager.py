import csv
from typing import List, Tuple

class WorkloadManager:
    """
    Handles loading and managing workload files.
    
    Responsible for:
    - Loading workload data from CSV files
    - Managing workload entries and filtering by time
    - Identifying models that should be injected at a given time
    
    Args:
        wl_file_path (str): Path to the workload CSV file
        blocking_age_threshold (int): The age at which a model blocks others if it can't be mapped.
    """
    def __init__(self, wl_file_path: str, blocking_age_threshold: int = 10):
        self.wl_file = wl_file_path
        self.blocking_age_threshold = blocking_age_threshold
        self.workload = []
        self.injected_models = set()  # Track models that have been injected
        
        # List of models ready to be injected.
        # Each entry is a dictionary: {'name': str, 'num_inputs': int, 'idx': int, 'ready_time': float, 'failure_age': int}
        self.ready_models = []
        
        self.last_time_checked = 0.0  # Last time get_models_to_inject was called
        self.load_workload()
        
    def load_workload(self) -> None:
        """
        Loads the workload from the specified CSV file.
        Each workload entry contains (inject_time_us, model, num_inputs, model_idx).
        """
        try:
            # Read the workload file
            with open(self.wl_file, 'r') as file:
                csv_reader = csv.DictReader(file)
                
                # Read each row from the CSV
                for row in csv_reader:
                    # Clean up the keys by stripping whitespace
                    cleaned_row = {k.strip(): v for k, v in row.items()}
                    
                    # Get model index from the file
                    model_idx = int(cleaned_row['net_idx'])
                    
                    # Get inject time which is already in microseconds
                    inject_time_us = float(cleaned_row['inject_time_us'])
                    
                    model = cleaned_row['network']
                    num_inputs = int(cleaned_row['num_inputs'])
                    
                    # Store as a tuple: (inject_time_us, model, num_inputs, model_idx)
                    self.workload.append((inject_time_us, model, num_inputs, model_idx))
                
                # Sort the workload by injection time
                self.workload.sort(key=lambda x: x[0])
                
                # Print the workload for debugging
                print("\n📋 Workload entries (sorted by injection time):")
                for i, (inject_time, model, num_inputs, model_idx) in enumerate(self.workload):
                    print(f"  {i+1}. Time: {inject_time:.2f} μs, Model: {model}, Inputs: {num_inputs}, Model Index: {model_idx}")
                    
        except FileNotFoundError:
            print(f"❌ ERROR: Workload file not found: {self.wl_file}")
            self.workload = []
        except Exception as e:
            print(f"❌ ERROR: Failed to load workload file: {e}")
            self.workload = []
    
    def increment_failure_age(self, model_idx: int) -> None:
        """
        Increments the failure age for a model that failed to be mapped.
        
        Args:
            model_idx (int): The index of the model that failed.
        """
        for model in self.ready_models:
            if model['idx'] == model_idx:
                model['failure_age'] += 1
                # print(f"INFO: Incremented failure age for model {model_idx} to {model['failure_age']}.")
                break
    
    def is_model_blocking(self, model_idx: int) -> bool:
        """
        Checks if a model's failure age has reached the blocking threshold.
        
        Args:
            model_idx (int): The index of the model to check.
            
        Returns:
            bool: True if the model is blocking, False otherwise.
        """
        for model in self.ready_models:
            if model['idx'] == model_idx:
                return model['failure_age'] >= self.blocking_age_threshold
        return False
    
    def update_ready_models(self, current_time_us: float) -> None:
        """
        Updates the list of models that are ready to be injected.
        
        Args:
            current_time_us (float): Current simulation time in microseconds
        """
        # Check the workload for models that have become eligible since the last check
        for entry in self.workload:
            inject_time_us, model_name, num_inputs, model_idx = entry
            
            # Skip models that have already been injected or are already in the ready list
            if model_idx in self.injected_models or any(ready['idx'] == model_idx for ready in self.ready_models):
                continue
                
            # If the model's inject time is less than or equal to the current global time,
            # add it to the ready list with its ready time and initial failure age.
            if inject_time_us <= current_time_us:
                self.ready_models.append({
                    'name': model_name,
                    'num_inputs': num_inputs,
                    'idx': model_idx,
                    'ready_time': inject_time_us,
                    'failure_age': 0
                })
    
    def get_models_to_inject(self, current_time_us: float) -> List[Tuple[str, int, int]]:
        """
        Returns models that need to be injected at the given time, using strict age-based ordering.
        
        Args:
            current_time_us (float): Current simulation time in microseconds
            
        Returns:
            List[Tuple[str, int, int]]: List of (model_name, num_inputs, model_idx) tuples
                                       for models to be injected, prioritized by age.
        """
        if not self.workload:
            return []
        
        # First, update the ready models list based on current time
        self.update_ready_models(current_time_us)
        
        # If there are no ready models, return an empty list
        if not self.ready_models:
            self.last_time_checked = current_time_us
            return []
            
        # Sort the ready models strictly by ready_time (oldest first)
        self.ready_models.sort(key=lambda x: x['ready_time'])
        
        # Return the full list of ready models, sorted by age
        prioritized_list = [(model['name'], model['num_inputs'], model['idx']) for model in self.ready_models]
        
        # Update the last time checked
        self.last_time_checked = current_time_us
        
        return prioritized_list
    
    def mark_model_injected(self, model_idx: int) -> None:
        """
        Mark a model as successfully injected.
        
        Args:
            model_idx (int): The index of the model to mark as injected
        """
        self.injected_models.add(model_idx)
        
        # Also remove from ready_models if present (just in case)
        self.ready_models = [n for n in self.ready_models if n['idx'] != model_idx]
    
    def get_remaining_count(self) -> int:
        """
        Returns the number of remaining models in the workload.
        
        Returns:
            int: Number of remaining models
        """
        return len(self.workload) - len(self.injected_models) 