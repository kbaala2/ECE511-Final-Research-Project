"""
Traffic Calculator Module

This module handles all traffic calculation logic for the chiplet simulator,
including activation communication and weight loading traffic. It provides
a clean separation of concerns by extracting traffic calculation from the
MappedModel class and providing access to system-level chiplet parameters.
"""

from typing import Dict, List, Optional

from .comm_types import (
    Phase, TrafficMatrixDict,
    ActivationCommPhase, WeightLoadingPhase
)
from .system import System


class TrafficCalculator:
    """
    Handles traffic calculation for different phase types in the chiplet simulator.
    
    This class encapsulates all traffic calculation logic and has access to the
    System object to query chiplet-specific parameters like bits_per_weight.
    """
    
    def __init__(self, 
                 system: System,
                 bits_per_activation: int = 8,
                 bits_per_packet: int = 128):
        """
        Initialize the TrafficCalculator with system information.
        
        Args:
            system: The System object containing chiplet configurations
            bits_per_activation: Number of bits per activation value
            bits_per_packet: Number of bits per network packet
        """
        self.system = system
        self.bits_per_activation = bits_per_activation
        self.bits_per_packet = bits_per_packet
    
    def calculate_traffic(self, 
                         phase: Phase,
                         model_def: dict,
                         model_metrics: List[dict],
                         mapping: List[tuple]) -> TrafficMatrixDict:
        """
        Calculate traffic for a given phase based on its type.
        This method is kept for backward compatibility but delegates to specific methods.
        
        Args:
            phase: The Phase object to calculate traffic for
            model_def: Model definition containing layer information
            model_metrics: List of layer metrics
            mapping: Layer-to-chiplet mapping
            
        Returns:
            TrafficMatrixDict containing the calculated traffic
        """
        if isinstance(phase, ActivationCommPhase):
            return self.calculate_activation_traffic(phase, model_def, model_metrics, mapping)
        elif isinstance(phase, WeightLoadingPhase):
            return self.calculate_weight_loading_traffic(phase, model_metrics, mapping)
        else:
            # Compute phases and unknown types don't have traffic
            return {}
    
    def calculate_activation_traffic(self,
                                    phase: ActivationCommPhase,
                                    model_def: dict,
                                    model_metrics: List[dict],
                                    mapping: List[tuple]) -> TrafficMatrixDict:
        """
        Calculate activation traffic between layers.
        
        Args:
            phase: The Phase object containing layer information
            model_def: Model definition with layer connectivity
            model_metrics: List of layer metrics
            mapping: Layer-to-chiplet mapping
            
        Returns:
            TrafficMatrixDict with activation traffic between chiplets
        """
        layer_idx = phase.layer_idx
        num_layers = len(model_def.get('layers', {}))
        
        if layer_idx >= num_layers - 1:
            return {}
        
        # Get the layer definition from the model
        if layer_idx not in model_def['layers']:
            print(f"❌ Invalid layer index {layer_idx} for traffic calculation")
            return {}
        
        layer_def = model_def['layers'][layer_idx]
        
        # Get receiving layers from the layer definition
        receiving_layers = layer_def.get('receiving_layers', [])
        
        # Skip if this layer has no receiving layers
        if not receiving_layers or receiving_layers == [-1]:
            return {}
        
        # Get the current layer metrics
        current_layer_metrics = None
        for layer_metrics in model_metrics:
            if layer_metrics.get('name') == layer_def.get('name', f"layer_{layer_idx}"):
                current_layer_metrics = layer_metrics
                break
        
        if not current_layer_metrics:
            print(f"❌ ERROR: Metrics not found for layer {layer_def.get('name', f'layer_{layer_idx}')}.")
            return {}
        
        # Use pre-calculated metrics
        total_activations = current_layer_metrics['output_activation']
        
        # Calculate total bits and packets
        total_activation_bits = total_activations * self.bits_per_activation
        total_packets = max(1, total_activation_bits // self.bits_per_packet)
        
        # Find current layer's chiplet mapping
        current_layer_mapping = None
        for map_layer_idx, chiplet_mappings in mapping:
            if map_layer_idx == layer_idx:
                current_layer_mapping = chiplet_mappings
                break
        
        if not current_layer_mapping:
            print(f"❌ ERROR: Missing mapping information for layer {layer_idx}")
            return {}
        
        # Initialize traffic dictionary
        traffic = {}
        
        # Calculate traffic to each receiving layer
        for receiving_layer_idx in receiving_layers:
            # Find receiving layer's chiplet mapping
            receiving_layer_mapping = None
            for map_layer_idx, chiplet_mappings in mapping:
                if map_layer_idx == receiving_layer_idx:
                    receiving_layer_mapping = chiplet_mappings
                    break
            
            if not receiving_layer_mapping:
                print(f"❌ ERROR: Missing mapping for receiving layer {receiving_layer_idx}")
                continue
            
            # For each chiplet pair between current and receiving layer, compute traffic
            for curr_chiplet_id, curr_pct in current_layer_mapping:
                # Initialize traffic for this source chiplet if not exists
                if curr_chiplet_id not in traffic:
                    traffic[curr_chiplet_id] = {}
                
                for recv_chiplet_id, recv_pct in receiving_layer_mapping:
                    if curr_chiplet_id != recv_chiplet_id:  # Only consider traffic between different chiplets
                        source_chiplet_output_packets = total_packets * (curr_pct / 100.0)
                        amount = max(1, int(source_chiplet_output_packets))
                        
                        # Accumulate traffic if this destination already exists
                        if recv_chiplet_id in traffic[curr_chiplet_id]:
                            traffic[curr_chiplet_id][recv_chiplet_id] += amount
                        else:
                            traffic[curr_chiplet_id][recv_chiplet_id] = amount
        
        return traffic
    
    def calculate_weight_loading_traffic(self,
                                        phase: WeightLoadingPhase,
                                        model_metrics: List[dict],
                                        mapping: List[tuple]) -> TrafficMatrixDict:
        """
        Calculate weight loading traffic from I/O to compute chiplets.
        Handles both single and combined weight loading phases.
        Uses chiplet-specific bits_per_weight values from the System.
        
        Args:
            phase: The WeightLoadingPhase object containing layer information
            model_metrics: List of layer metrics
            mapping: Layer-to-chiplet mapping
            
        Returns:
            TrafficMatrixDict with weight loading traffic from I/O chiplets
        """
        # Handle combined weight loading
        if phase.is_combined():
            combined_traffic = {}
            
            for layer_idx in phase.layers_to_load:
                layer_traffic = self._calculate_single_layer_weight_traffic(
                    layer_idx, model_metrics, mapping
                )
                # Merge layer traffic into combined traffic
                for src, dests in layer_traffic.items():
                    if src not in combined_traffic:
                        combined_traffic[src] = {}
                    for dst, amount in dests.items():
                        if dst in combined_traffic[src]:
                            combined_traffic[src][dst] += amount
                        else:
                            combined_traffic[src][dst] = amount
            
            return combined_traffic
        else:
            # Single layer weight loading
            layer_idx = phase.layers_to_load[0] if phase.layers_to_load else -1
            return self._calculate_single_layer_weight_traffic(
                layer_idx, model_metrics, mapping
            )
    
    def _calculate_single_layer_weight_traffic(self,
                                              layer_idx: int,
                                              model_metrics: List[dict],
                                              mapping: List[tuple]) -> TrafficMatrixDict:
        """
        Calculate weight loading traffic for a single layer.
        
        Args:
            layer_idx: Index of the layer
            model_metrics: List of layer metrics
            mapping: Layer-to-chiplet mapping
            
        Returns:
            TrafficMatrixDict with weight loading traffic for this layer
        """
        # Get layer metrics to find weight amount
        layer_metrics = None
        for metrics in model_metrics:
            layer_name = f"layer_{layer_idx}"
            if metrics.get('name') == layer_name:
                layer_metrics = metrics
                break
        
        if not layer_metrics:
            print(f"❌ ERROR: Metrics not found for layer {layer_idx}")
            print(f"  Available metrics: {[m.get('name', 'UNNAMED') for m in model_metrics]}")
            return {}
        
        num_weights = layer_metrics.get('num_weights', 0)
        print(f"  Found {num_weights} weights for layer {layer_idx}")
        if num_weights == 0:
            print(f"  No weights to load, returning empty traffic")
            return {}
        
        # Get chiplet mapping for this layer
        layer_mapping = None
        for map_layer_idx, chiplet_mappings in mapping:
            if map_layer_idx == layer_idx:
                layer_mapping = chiplet_mappings
                break
        
        if not layer_mapping:
            print(f"❌ ERROR: Missing mapping for layer {layer_idx}")
            print(f"  Available mappings for layers: {[m[0] for m in mapping]}")
            return {}
        
        print(f"  Layer {layer_idx} mapped to chiplets: {[(c, p) for c, p in layer_mapping]}")
        
        # Check if I/O chiplets exist in the system
        if not self.system.io_chiplet_ids:
            print(f"❌ ERROR: No I/O chiplets found in system for weight loading")
            return {}
        
        print(f"  I/O chiplets available: {self.system.io_chiplet_ids}")
        traffic = {}
        
        # Distribute weights to compute chiplets based on their allocation
        for chiplet_id, percentage in layer_mapping:
            # Throw an error if an I/O chiplet is found in the layer mapping (shouldn't happen)
            if self.system.is_io_chiplet(chiplet_id):
                raise ValueError(f"ERROR: I/O chiplet {chiplet_id} found in layer mapping for layer {layer_idx}. This should not occur. Please check the network mapping logic.")
            
            # Get the actual bits_per_weight from the destination chiplet
            if chiplet_id <= self.system.num_chiplets:
                chiplet = self.system.chiplets[chiplet_id - 1]  # Adjust for 0-based indexing
                bits_per_weight = chiplet.bits_per_weight
            else:
                print(f"❌ ERROR: Invalid chiplet ID {chiplet_id}")
                continue
            
            # Calculate weights and packets for this chiplet
            weights_for_chiplet = int(num_weights * (percentage / 100.0))
            if weights_for_chiplet == 0:
                continue
                
            total_weight_bits = weights_for_chiplet * bits_per_weight
            packets_for_chiplet = max(1, total_weight_bits // self.bits_per_packet)
            
            # Determine which I/O chiplet serves this compute chiplet (closest by network distance)
            io_chiplet = self.system.get_io_chiplet_for_compute(chiplet_id)
            if io_chiplet is None:
                print(f"❌ ERROR: Could not find I/O chiplet for compute chiplet {chiplet_id}")
                continue
            
            # Add traffic from I/O chiplet to compute chiplet
            if io_chiplet not in traffic:
                traffic[io_chiplet] = {}
            
            traffic[io_chiplet][chiplet_id] = packets_for_chiplet
            
            # Log the weight loading details for debugging
            print(f"  📦 Weight loading for layer {layer_idx}, chiplet {chiplet_id}: "
                  f"{weights_for_chiplet} weights × {bits_per_weight} bits/weight = "
                  f"{packets_for_chiplet} packets from I/O chiplet {io_chiplet}")
        
        return traffic
