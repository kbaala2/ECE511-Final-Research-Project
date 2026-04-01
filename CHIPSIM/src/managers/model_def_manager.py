import os
import importlib.util
import glob
import re
import yaml
import json
from typing import Dict, Any, List

class ModelDefinitionManager:
    """
    Handles loading and managing model definitions.
    
    Responsible for:
    - Loading model definitions from Python files
    - Providing access to network structure and layer information
    
    Args:
        network_definitions_file (str): Path to model definitions file
    """
    def __init__(self, model_definitions_file: str):
        self.model_definitions_file = model_definitions_file
        self.models = self._load_definitions()
        
    def _load_definitions(self) -> Dict[str, Any]:
        """
        Load model definitions.
        
        Returns:
            Dict[str, Any]: Dictionary containing model definitions
        """
        return self._load_from_file()
            
    def _load_from_file(self) -> Dict[str, Any]:
        """
        Load model definitions from the specified Python file.
        
        Returns:
            Dict[str, Any]: Dictionary containing model definitions.
        """
        try:
            # Get the absolute path to the model definitions file
            abs_path = os.path.abspath(self.model_definitions_file)
            
            # Load the module from the file path
            spec = importlib.util.spec_from_file_location("model_definitions", abs_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the models dictionary from the module - first try MODEL_DEFINITIONS (new format)
            # then fall back to models (old format) if not found
            models = getattr(module, 'MODEL_DEFINITIONS', None)
            if models is None:
                models = getattr(module, 'models', {})
                
            # If models is empty, throw an error  
            if not models:
                raise ValueError(f"❌ No model definitions loaded from {abs_path}!")
            
            print(f"✅ Loaded {len(models)} model definitions from file: {abs_path}")
            
            # Return the models dictionary from the module
            return models
        except Exception as e:
            print(f"❌ Error loading model definitions from file: {e}")
            return {}
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get a specific model definition by name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model definition or empty dict if not found
        """
        return self.models.get(model_name, {})
    
    def get_model_names(self) -> List[str]:
        """
        Get a list of all available model names.
        
        Returns:
            List[str]: List of model names
        """
        return list(self.models.keys())