"""
Model Processor Module

Handles conversion of model definitions into simulation-ready layer metrics.
Calculates fundamental metrics like MACs, activations, and weight counts for
different layer types (Conv2d, Conv1D, Linear, Self-Attention).
"""

from typing import List, Dict, Any, Optional


class ModelProcessor:
    """
    Processes model definitions into layer metrics for simulation.
    
    This class analyzes model definitions and calculates key metrics needed
    for compute simulation and resource allocation, including:
    - Input/output activations
    - Total MACs (multiply-accumulate operations)
    - Weight counts
    - Crossbar requirements
    """
    
    def __init__(self):
        """Initialize the ModelProcessor."""
        pass
    
    def process_model(self, model_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the model definition and return the model metrics.
        
        Args:
            model_def (dict): Model definition with 'layers' dictionary
            
        Returns:
            list: List of layer metric dictionaries
            
        Raises:
            ValueError: If model definition is invalid or contains unsupported layer types
        """
        
        # List to hold layer information
        model_metrics = []
        
        # Expect new format: model_def['layers'] is a dict mapping idx -> layer dict
        layers_dict = model_def.get('layers', None)
        if not isinstance(layers_dict, dict) or len(layers_dict) == 0:
            raise ValueError("Model definition must contain a non-empty 'layers' dictionary in the new format.")

        # Sort layers by numeric index (keys may be int or str)
        try:
            layer_indices = sorted(layers_dict.keys(), key=lambda k: int(k))
        except Exception:
            layer_indices = sorted(layers_dict.keys())
        
        # Track dimensions for convolutional layers
        current_input_height = None
        current_input_width = None
        
        # Process each layer
        for idx in layer_indices:
            layer = layers_dict[idx]
            params = layer['parameters']
            description = layer['description']
            
            # Extract common parameters
            try:
                name = f"layer_{int(idx)}"
            except Exception:
                name = f"layer_{idx}"
            input_channels = params['input_channels']
            output_channels = params['output_channels']
            
            # Process based on layer type
            if 'Conv2d' in description:
                layer_info = self._process_conv2d_layer(
                    idx, name, params, 
                    current_input_height, current_input_width
                )
                # Update dimensions for next layer
                current_input_height = params['output_height']
                current_input_width = params['output_width']
                
            elif 'Conv1D' in description:
                layer_info = self._process_conv1d_layer(
                    idx, name, params,
                    current_input_height, current_input_width
                )
                # Update dimensions for next layer
                current_input_height = params['output_height']
                current_input_width = 1
                
            elif 'Linear' in description:
                layer_info = self._process_linear_layer(idx, name, params)
                
            elif 'DPTViTSelfAttention' in description or 'SelfAttention' in description or 'ViTSelfAttention' in description:
                layer_info = self._process_self_attention_layer(idx, name, params)
                
            else:
                # Raise an error for unsupported layer types
                raise ValueError(f"Unsupported layer type: {description}")
            
            # Add to model metrics
            model_metrics.append(layer_info)
        
        return model_metrics
    
    def _process_conv2d_layer(self, idx, name, params, current_input_height, current_input_width) -> Dict[str, Any]:
        """
        Process a Conv2d layer and calculate its metrics.
        
        Args:
            idx: Layer index
            name: Layer name
            params: Layer parameters
            current_input_height: Current input height from previous layer
            current_input_width: Current input width from previous layer
            
        Returns:
            dict: Layer metrics
        """
        layer_type = 'conv'
        
        # Initialize input dimensions if not set
        if current_input_height is None:
            current_input_height = params['input_height'] * params.get('output_height', 1)
            current_input_width = params['input_width'] * params.get('output_width', 1)
        
        # Get convolution parameters
        input_channels = params['input_channels']
        output_channels = params['output_channels']
        kernel_size = params['weight_height']  # Assuming square kernels
        stride = params.get('height_stride', 1)
        padding = params.get('padding', 0)
        
        # Output dimensions
        output_height = params['output_height']
        output_width = params['output_width']
        
        # Calculate metrics
        input_activation = current_input_height * current_input_width * input_channels
        total_macs = output_channels * output_height * output_width * (kernel_size ** 2) * input_channels
        num_weights = output_channels * input_channels * (kernel_size ** 2)
        filter_size = input_channels * (kernel_size ** 2)
        output_activation = output_height * output_width * output_channels
        
        # Create layer info dictionary
        return {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,
            'crossbars_required': num_weights,
            'filter_size': filter_size,
            'out_channels': output_channels,
            'output_activation': output_activation,
        }
    
    def _process_conv1d_layer(self, idx, name, params, current_input_height, current_input_width) -> Dict[str, Any]:
        """
        Process a Conv1D layer and calculate its metrics.
        
        Args:
            idx: Layer index
            name: Layer name
            params: Layer parameters
            current_input_height: Current input height from previous layer
            current_input_width: Current input width from previous layer
            
        Returns:
            dict: Layer metrics
        """
        layer_type = 'conv1d'
        
        # Initialize input dimensions if not set
        if current_input_height is None:
            current_input_height = params['input_height'] * params.get('output_height', 1)
            current_input_width = 1  # Width is 1 for Conv1D
        
        # Get convolution parameters
        input_channels = params['input_channels']
        output_channels = params['output_channels']
        kernel_size = params['weight_height']  # Filter length
        stride = params.get('height_stride', 1)
        padding = params.get('padding', 0)
        
        # Output dimensions
        output_height = params['output_height']
        output_width = 1  # Width is 1 for Conv1D
        
        # Calculate metrics
        input_activation = current_input_height * input_channels
        total_macs = output_channels * output_height * kernel_size * input_channels
        num_weights = output_channels * input_channels * kernel_size
        filter_size = input_channels * kernel_size
        output_activation = output_height * output_channels
        
        # Create layer info dictionary
        return {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,
            'crossbars_required': num_weights,
            'filter_size': filter_size,
            'out_channels': output_channels,
            'output_activation': output_activation,
        }
    
    def _process_linear_layer(self, idx, name, params) -> Dict[str, Any]:
        """
        Process a Linear (fully connected) layer and calculate its metrics.
        
        Args:
            idx: Layer index
            name: Layer name
            params: Layer parameters
            
        Returns:
            dict: Layer metrics
        """
        layer_type = 'fc'
        
        # Linear layer metrics
        in_features = params['input_channels']
        out_features = params['output_channels']
        
        # Check for P dimension (sequence length) in output_height
        # This is critical for transformer models where Linear layers process sequences
        p_dimension = params.get('output_height', 1)
        
        input_activation = in_features * p_dimension
        total_macs = out_features * in_features * p_dimension  # Include P dimension!
        num_weights = out_features * in_features
        output_activation = out_features * p_dimension
        
        # Create layer info dictionary
        return {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,
            'crossbars_required': num_weights,
            'in_features': in_features,
            'out_features': out_features,
            'output_activation': output_activation,
        }
    
    def _process_self_attention_layer(self, idx, name, params) -> Dict[str, Any]:
        """
        Process a Self-Attention layer and calculate its metrics.
        
        Self-attention operations are matrix multiplications without learnable weights:
        - ViTSelfAttention_QK: Q@K^T operation [P×C] @ [C×M]
        - ViTSelfAttention_AttnV: Attn@V operation [P×C] @ [C×M]
        
        Args:
            idx: Layer index
            name: Layer name
            params: Layer parameters
            
        Returns:
            dict: Layer metrics
        """
        layer_type = 'self_attention'
        
        # Self-attention now uses P dimension for proper matrix multiplication representation
        # Format: {C: inner_dim, M: output_dim, P: batch/sequence_dim}
        # For ViTSelfAttention_QK: {C: 768, M: 197, P: 197} - Q@K^T: [197x768] @ [768x197]
        # For ViTSelfAttention_AttnV: {C: 197, M: 768, P: 197} - Attn@V: [197x197] @ [197x768]
        
        c_dim = params.get('input_channels', 768)  # Inner dimension
        m_dim = params.get('output_channels', 197)  # Output dimension  
        p_dim = params.get('output_height', 197)   # Batch/sequence dimension
        
        # Input and output activations
        input_activation = p_dim * c_dim
        output_activation = p_dim * m_dim
        
        # Self-attention MACs: Matrix multiplication [P×C] @ [C×M] = P×C×M
        total_macs = p_dim * c_dim * m_dim
        
        # Attention operations have NO learnable weights
        # These are pure matrix multiplications using pre-computed Q, K, V from fused QKV layer
        num_weights = 0  # No weights to load from I/O
        
        # However, they still require crossbars for computation
        # Allocate based on the matrix multiplication size
        crossbars_required = c_dim * m_dim  # Resources needed for computation
        
        # Create layer info dictionary
        return {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,  # No weights to load
            'crossbars_required': crossbars_required,  # Resources for computation
            'out_channels': m_dim,
            'seq_length': p_dim,
            'output_activation': output_activation,
        }

