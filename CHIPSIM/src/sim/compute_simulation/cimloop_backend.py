"""
CIMLoop Backend for Compute Simulation

Handles compute simulation for In-Memory Computing (IMC) chiplets using the CIMLoop API.
Generates YAML configurations and runs simulations through the CIMLoop framework.
"""

import os
import re
import time
from integrations.CIMLoop_API import CIMLoop_API


class CIMLoopBackend:
    """
    Backend for simulating IMC chiplets using CIMLoop.
    """
    
    def __init__(self):
        """Initialize the CIMLoop backend."""
        self.cimloop_api = CIMLoop_API()
    
    def generate_yaml_config(self, model_name, layer_idx, chiplet_id, partitioned_layer, num_layers):
        """
        Generate a YAML configuration for a model chunk based on the original template.
        
        Args:
            model_name (str): Name of the model (e.g., 'ResNet18').
            layer_idx (int): Index of the layer in the model.
            chiplet_id (int): ID of the chiplet.
            partitioned_layer (dict): Partitioned layer definition with 'description' and 'parameters' fields.
            num_layers (int): Total number of layers in the model.
            
        Returns:
            str: YAML configuration content (instead of file path).
            
        Raises:
            ValueError: If required parameters are missing or if Conv1d layer type is encountered.
            FileNotFoundError: If template file does not exist.
            IOError: If there's an error reading the template file.
        """     
        # Get the template file
        # Determine padding based on total number of layers
        try:
            # Calculate the necessary padding width based on total layers
            padding_width = len(str(num_layers))
            # Use string formatting with dynamic padding width
            formatted_layer_idx = f"{layer_idx:0{padding_width}d}"
            # Go up 4 levels: compute_simulation -> sim -> src -> project_root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            template_file = os.path.join(project_root, "assets", "DNN_models", 
                                         model_name.lower(), 
                                         f"{formatted_layer_idx}.yaml")
            
            # Check if the file exists with the calculated padding
            if not os.path.exists(template_file) and padding_width > 1:
                # Try without padding as fallback
                template_file = os.path.join(project_root, "assets", "DNN_models", 
                                            model_name.lower(), 
                                            f"{layer_idx}.yaml")
        except ValueError as e:
            raise ValueError(f"Error formatting layer index: {e}")
        
        # Check if the template file exists
        if not os.path.exists(template_file):
            error_msg = f"Template file {template_file} does not exist for layer {layer_idx}."
            print(f"Error: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Read the template file
        try:
            with open(template_file, 'r') as f:
                template_content = f.read()
        except Exception as e:
            error_msg = f"Error reading template file {template_file}: {e}"
            print(f"Error: {error_msg}")
            raise IOError(error_msg)
        
        # Fix the include path to point to the correct location for problem_base.yaml
        # The template file may use ../problem_base.yaml but we need to use workloads/problem_base.yaml
        template_content = template_content.replace(
            "{{include_text('../problem_base.yaml')}}", 
            "{{include_text('workloads/problem_base.yaml')}}"
        )
        
        # Modify the template content based on the partitioned layer
        # 1. Keep the include directive for problem_base.yaml
        # 2. Update the instance parameters based on the partitioned layer
        
        # Get layer description and parameters
        if 'description' not in partitioned_layer:
            raise ValueError(f"Missing 'description' field in partitioned layer for layer {layer_idx}")
        
        if 'parameters' not in partitioned_layer:
            raise ValueError(f"Missing 'parameters' field in partitioned layer for layer {layer_idx}")
            
        layer_description = partitioned_layer['description']
        layer_params = partitioned_layer['parameters']
        
        # Check for percentage field, this is optional so we'll still use a default if not provided
        percentage = partitioned_layer.get('percentage', 100)
        
        # Check for Conv1d layer type and throw error
        if 'Conv1d' in layer_description:
            error_msg = f"Conv1d layer type is not supported (layer {layer_idx})"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Replace the instance line with updated parameters
        if 'Conv2d' in layer_description:  # Convolutional layer
            # Get the parameters for the instance line
            if 'input_channels' not in layer_params:
                raise ValueError(f"Missing 'input_channels' parameter for Conv2d layer {layer_idx}")
            if 'output_channels' not in layer_params:
                raise ValueError(f"Missing 'output_channels' parameter for Conv2d layer {layer_idx}")
            if 'weight_height' not in layer_params:
                raise ValueError(f"Missing 'weight_height' parameter for Conv2d layer {layer_idx}")
                
            in_channels = layer_params['input_channels']
            out_channels = layer_params['output_channels']
            kernel_size = layer_params['weight_height']  # Assuming weight_height = weight_width for square kernels
            stride = layer_params.get('HStride', 1)  # Default stride to 1 if not specified
            
            # Calculate output dimensions based on the original template
            # Extract them from the template using regex
            instance_match = re.search(r'instance:\s*{([^}]*)}', template_content)
            if instance_match:
                instance_str = instance_match.group(1)
                # Extract P and Q (output height and width)
                p_match = re.search(r'P:\s*(\d+)', instance_str)
                q_match = re.search(r'Q:\s*(\d+)', instance_str)
                
                if p_match:
                    p_value = int(p_match.group(1))
                elif 'output_height' in layer_params:
                    p_value = layer_params['output_height']
                else:
                    raise ValueError(f"Cannot determine output height (P) for Conv2d layer {layer_idx}")
                    
                if q_match:
                    q_value = int(q_match.group(1))
                elif 'output_width' in layer_params:
                    q_value = layer_params['output_width']
                else:
                    raise ValueError(f"Cannot determine output width (Q) for Conv2d layer {layer_idx}")
                
                # Update the instance line with new parameters
                new_instance = f"instance: {{C: {in_channels}, M: {out_channels}, " \
                               f"P: {p_value}, Q: {q_value}, " \
                               f"R: {kernel_size}, S: {kernel_size}, " \
                               f"HStride: {stride}, WStride: {stride}}}"
                
                # Replace the instance line in the template
                template_content = re.sub(r'instance:\s*{[^}]*}', new_instance, template_content)
            else:
                raise ValueError(f"Cannot find instance definition in template for Conv2d layer {layer_idx}")
                
            # Add a note indicating this is a partitioned convolutional layer for a specific chiplet
            # Use regex to find and replace any notes field (more flexible than exact match)
            notes_pattern = r'notes:\s*[^\n]*'
            if not re.search(notes_pattern, template_content):
                raise ValueError(f"Cannot find 'notes:' field in template for Conv2d layer {layer_idx}")
                
            template_content = re.sub(notes_pattern, 
                                    f"notes: Conv2d - Partitioned for Chiplet {chiplet_id} ({percentage}%)", 
                                    template_content)
                
        elif 'Linear' in layer_description:  # Fully connected layer
            # For FC layers, check if there's a P dimension (sequence length for transformers)
            if 'input_channels' not in layer_params:
                raise ValueError(f"Missing 'input_channels' parameter for Linear layer {layer_idx}")
            if 'output_channels' not in layer_params:
                raise ValueError(f"Missing 'output_channels' parameter for Linear layer {layer_idx}")
                
            in_features = layer_params['input_channels']
            out_features = layer_params['output_channels']
            
            # Check for P dimension (output_height) - critical for transformer models
            p_dimension = layer_params.get('output_height', 1)
            
            # Update the instance line with new parameters
            if p_dimension > 1:
                # Include P dimension for transformer Linear layers
                new_instance = f"instance: {{C: {in_features}, M: {out_features}, P: {p_dimension}}}"
            else:
                # Traditional FC layer without P dimension
                new_instance = f"instance: {{C: {in_features}, M: {out_features}}}"
            
            # Replace the instance line in the template
            if not re.search(r'instance:\s*{[^}]*}', template_content):
                raise ValueError(f"Cannot find instance definition in template for Linear layer {layer_idx}")
                
            template_content = re.sub(r'instance:\s*{[^}]*}', new_instance, template_content)
            
            # Add a note indicating this is a partitioned fully connected layer for a specific chiplet
            # Use regex to find and replace any notes field (more flexible than exact match)
            notes_pattern = r'notes:\s*[^\n]*'
            if not re.search(notes_pattern, template_content):
                raise ValueError(f"Cannot find 'notes:' field in template for Linear layer {layer_idx}")
                
            template_content = re.sub(notes_pattern, 
                                    f"notes: Linear - Partitioned for Chiplet {chiplet_id} ({percentage}%)", 
                                    template_content)
        elif 'DPTViTSelfAttention' in layer_description or 'SelfAttention' in layer_description or 'ViTSelfAttention' in layer_description:
            # Self-attention layer (from transformer models)
            # These layers now use P dimension for semantically correct matrix multiplication
            # The templates have been updated to use {C: X, M: Y, P: Z} format
            
            # We keep the template's instance parameters as-is (they now use P dimension)
            # Just verify the template has an instance definition
            if not re.search(r'instance:\s*{[^}]*}', template_content):
                raise ValueError(f"Cannot find instance definition in template for self-attention layer {layer_idx}")
            
            # DO NOT replace the instance line - keep the P-dimension values from the template!
            
            # Add a note indicating this is a partitioned self-attention layer for a specific chiplet
            if "notes: From einsum" in template_content:
                template_content = template_content.replace("notes: From einsum", 
                                                          f"notes: Self-Attention - Partitioned for Chiplet {chiplet_id} ({percentage}%)")
            elif "notes: DPTViTSelfAttention" in template_content:
                template_content = template_content.replace("notes: DPTViTSelfAttention", 
                                                          f"notes: Self-Attention - Partitioned for Chiplet {chiplet_id} ({percentage}%)")
            elif "notes: Attention" in template_content:
                # Handle the new format with P dimension
                template_content = re.sub(r'notes: Attention[^\n]*', 
                                        f"notes: Self-Attention - Partitioned for Chiplet {chiplet_id} ({percentage}%)", 
                                        template_content)
            else:
                # Try to find any notes field and update it
                template_content = re.sub(r'notes:\s*[^\n]*', 
                                        f"notes: Self-Attention - Partitioned for Chiplet {chiplet_id} ({percentage}%)", 
                                        template_content)
        else:
            raise ValueError(f"Unsupported layer type '{layer_description}' for layer {layer_idx}")
        
        return template_content
    
    def simulate(self, yaml_config, chiplet_id, chiplet_type, batch_size=1):
        """
        Simulate compute using CIMLoop API.
        
        Args:
            yaml_config (str): YAML configuration for the model chunk
            chiplet_id (int): ID of the chiplet to simulate on
            chiplet_type (str): Type of the chiplet
            batch_size (int): Batch size for simulation
            
        Returns:
            dict: Simulation results containing latency_us, energy_fj, cycles
        """
        print(f"🔄 Making CIMLoop API request for chiplet {chiplet_id}...")
        
        # Run the simulation using CimLoop API with a single YAML config
        simulation_result = self._run_cimloop_simulation(
            yaml_configs=[yaml_config],
            batch_size=batch_size,
            chiplet_type=chiplet_type
        )
        
        if simulation_result is None:
            print(f"❌ CimLoop API returned no results (None)")
            raise RuntimeError("CimLoop API returned None")
        
        # Extract results
        latency_seconds = simulation_result.get('total_runtime_seconds', 0)
        energy_fj = simulation_result.get('total_energy_fJ', 0)
        cycles = simulation_result.get('total_cycles', 0)
        
        # Convert latency from seconds to microseconds
        latency_us = latency_seconds * 1e6
        
        # Log the successful results
        print(f"✅ CIMLoop simulation results: Latency={latency_us:,.2f} μs, Energy={energy_fj:,.2f} fJ")
        
        # Create the result dictionary
        result = {
            'latency_us': latency_us,
            'energy_fj': energy_fj,
            'cycles': cycles
        }
        
        return result
    
    def _run_cimloop_simulation(self, yaml_configs, batch_size, chiplet_type):
        """
        Run a simulation using the CimLoop API.
        This is a wrapper around the actual API call.
        
        Args:
            yaml_configs (list): List of YAML configuration strings
            batch_size (int): Batch size for simulation
            chiplet_type (str): Type of the chiplet being simulated
            
        Returns:
            dict: Simulation results from CimLoop
        """
        return self.cimloop_api.run_simulation_with_yaml(
            yaml_configs=yaml_configs,
            batch_size=batch_size,
            system_config=None,
            chiplet_type=chiplet_type,
            debug=False
        )

