from typing import Dict, List, Tuple, Any

class LayerPartitioner:
    """
    Handles partitioning of network layers across chiplets.
    
    Responsible for:
    - Partitioning layers based on mapping distribution
    - Adjusting for rounding errors to ensure correct partitioning
    - Verifying partitioning integrity
    """
    
    def __init__(self):
        """Initialize the LayerPartitioner."""
        pass
        
    def partition_layer(self, layer_def: Dict[str, Any], chiplet_mappings: List[Tuple[int, float]]) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Partition a single layer based on the mapping distribution.
        
        Args:
            layer_def (Dict[str, Any]): Original layer definition from network definitions file
                                  (e.g., {'description': 'Conv2d layer', 'parameters': {...}}).
            chiplet_mappings (List[Tuple[int, float]]): List of tuples containing (chiplet_id, percentage).
            
        Returns:
            List[Tuple[int, Dict[str, Any]]]: List of tuples (chiplet_id, partitioned_layer_def) 
                                        containing the partitioned layer definitions.
        """
        # List to store partitioned layer definitions for each chiplet
        network_chunks = []
        
        # Determine layer type
        description = layer_def.get('description', '')
        is_conv_layer = 'Conv2d' in description or 'Conv1D' in description
        is_attention_layer = 'DPTViTSelfAttention' in description or 'SelfAttention' in description or 'ViTSelfAttention' in description
        params = layer_def.get('parameters', {})
        
        # For each chiplet that has a portion of this layer
        for chiplet_id, percentage in chiplet_mappings:
            # Create a copy of the original layer definition
            partitioned_layer = layer_def.copy()
            
            # Make a copy of the parameters to modify
            partitioned_params = params.copy()
            partitioned_layer['parameters'] = partitioned_params
            
            # Add a suffix to the layer name to indicate it's a partition
            partitioned_layer['name'] = f"{layer_def.get('name', layer_def.get('description', 'layer'))}_chiplet{chiplet_id}"
            
            # Add percentage information for reference
            partitioned_layer['percentage'] = percentage
            
            # Propagate and partition total_macs proportionally if it exists
            if 'total_macs' in layer_def:
                original_total_macs = layer_def['total_macs']
                partitioned_layer['total_macs'] = original_total_macs * (percentage / 100.0)
            
            # Adjust the layer parameters based on the percentage and layer type
            if is_conv_layer:
                # For convolutional layers, partition along the output channels
                original_out_channels = params.get('output_channels', 1)
                partitioned_out_channels = max(1, round(original_out_channels * percentage / 100))
                partitioned_params['output_channels'] = partitioned_out_channels
            elif is_attention_layer:
                # For self-attention layers, partition along the output_channels (dimension per head)
                # This maintains head coherence while distributing computation
                original_out_channels = params.get('output_channels', 1)
                partitioned_out_channels = max(1, round(original_out_channels * percentage / 100))
                partitioned_params['output_channels'] = partitioned_out_channels
            else:
                # For fully connected layers, partition along the output features
                original_out_features = params.get('output_channels', 1)
                partitioned_out_features = max(1, round(original_out_features * percentage / 100))
                partitioned_params['output_channels'] = partitioned_out_features
            
            # Store the partitioned layer with its chiplet ID
            network_chunks.append((chiplet_id, partitioned_layer))
        
        # Adjust for any rounding errors
        self._adjust_for_rounding_errors(layer_def, network_chunks)
        
        # Final verification check to ensure the sum of partitioned channels equals the original
        param_key = 'output_channels'
        original_value = params.get(param_key, 1)
            
        partitioned_sum = sum(chunk[1]['parameters'][param_key] for chunk in network_chunks)
        
        if partitioned_sum != original_value:
            print(f"Warning: After adjustment, the sum of partitioned {param_key} ({partitioned_sum}) " 
                  f"does not equal the original value ({original_value}) for layer {layer_def.get('name', 'unknown')}. "
                  f"This should not happen and indicates a bug in the adjustment algorithm.")
        
        return network_chunks

    def _adjust_for_rounding_errors(self, original_layer: Dict[str, Any], network_chunks: List[Tuple[int, Dict[str, Any]]]):
        """
        Adjust partitioned layers to account for rounding errors.
        
        Args:
            original_layer (Dict[str, Any]): Original layer definition.
            network_chunks (List[Tuple[int, Dict[str, Any]]]): List of tuples (chiplet_id, partitioned_layer_def).
        """
        # Determine which parameter to adjust based on layer type
        description = original_layer.get('description', '')
        is_conv_layer = 'Conv2d' in description or 'Conv1D' in description
        is_attention_layer = 'DPTViTSelfAttention' in description or 'SelfAttention' in description or 'ViTSelfAttention' in description
        params = original_layer.get('parameters', {})
        
        # Determine the parameter key based on layer type
        param_key = 'output_channels'
        original_value = params.get(param_key, 1)
        
        # Calculate the sum of partitioned values
        partitioned_sum = sum(chunk[1]['parameters'][param_key] for chunk in network_chunks)
        
        # If there's a discrepancy, adjust the partitions
        if partitioned_sum != original_value:
            difference = original_value - partitioned_sum
            
            if difference > 0:
                # Too few output channels/features, increase the smallest partition
                self._handle_too_few_case(network_chunks, param_key, difference)
            else:
                # Too many output channels/features, reduce the largest partition
                self._handle_too_many_case(network_chunks, param_key, difference)
    
    def _handle_too_few_case(self, network_chunks: List[Tuple[int, Dict[str, Any]]], param_key: str, difference: int):
        """
        Handle the case where the sum of partitioned values is less than the original value.
        
        Args:
            network_chunks (List[Tuple[int, Dict[str, Any]]]): List of tuples (chiplet_id, partitioned_layer_def).
            param_key (str): The parameter key to adjust ('output_channels').
            difference (int): The difference to distribute.
        """
        # Find the smallest partition
        min_chunk_idx = None
        min_value = float('inf')
        
        for i, (chiplet_id, partitioned_layer) in enumerate(network_chunks):
            if partitioned_layer['parameters'][param_key] < min_value:
                min_value = partitioned_layer['parameters'][param_key]
                min_chunk_idx = i
        
        # Adjust the smallest partition
        if min_chunk_idx is not None:
            network_chunks[min_chunk_idx][1]['parameters'][param_key] += difference
    
    def _handle_too_many_case(self, network_chunks: List[Tuple[int, Dict[str, Any]]], param_key: str, difference: int):
        """
        Handle the case where the sum of partitioned values is greater than the original value.
        
        Args:
            network_chunks (List[Tuple[int, Dict[str, Any]]]): List of tuples (chiplet_id, partitioned_layer_def).
            param_key (str): The parameter key to adjust ('output_channels').
            difference (int): The difference to distribute (negative value).
            
        Note: difference is expected to be negative in this case
        """
        # Find the largest partition
        max_chunk_idx = None
        max_value = 0
        
        for i, (chiplet_id, partitioned_layer) in enumerate(network_chunks):
            if partitioned_layer['parameters'][param_key] > max_value:
                max_value = partitioned_layer['parameters'][param_key]
                max_chunk_idx = i
        
        # Adjust the largest partition
        if max_chunk_idx is not None:
            # Ensure we don't reduce below 1
            new_value = max(1, network_chunks[max_chunk_idx][1]['parameters'][param_key] + difference)
            network_chunks[max_chunk_idx][1]['parameters'][param_key] = new_value
            
            # If we couldn't reduce enough (because we hit the minimum of 1),
            # distribute the remaining difference to other partitions
            remaining_difference = difference - (new_value - max_value)
            if remaining_difference < 0:
                self._distribute_remaining_difference(network_chunks, param_key, max_chunk_idx, remaining_difference)
    
    def _distribute_remaining_difference(self, network_chunks: List[Tuple[int, Dict[str, Any]]], 
                                        param_key: str, excluded_idx: int, remaining_difference: int):
        """
        Distribute remaining difference among other partitions when the largest partition can't be reduced enough.
        
        Args:
            network_chunks (List[Tuple[int, Dict[str, Any]]]): List of tuples (chiplet_id, partitioned_layer_def).
            param_key (str): The parameter key to adjust ('output_channels').
            excluded_idx (int): Index of the chunk to exclude from distribution.
            remaining_difference (int): The remaining difference to distribute (negative value).
        """
        # Sort partitions by size (largest first) excluding the one we just adjusted
        sorted_chunks = [(i, chunk) for i, chunk in enumerate(network_chunks) if i != excluded_idx]
        sorted_chunks.sort(key=lambda x: x[1][1]['parameters'][param_key], reverse=True)
        
        # Distribute the remaining difference
        for i, (chunk_idx, (chiplet_id, partitioned_layer)) in enumerate(sorted_chunks):
            if remaining_difference == 0:
                break
                
            # Calculate how much we can reduce this partition
            can_reduce = min(partitioned_layer['parameters'][param_key] - 1, abs(remaining_difference))
            if can_reduce > 0:
                network_chunks[chunk_idx][1]['parameters'][param_key] -= can_reduce
                remaining_difference += can_reduce 