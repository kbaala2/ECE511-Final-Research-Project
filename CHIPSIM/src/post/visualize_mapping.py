#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class ChipletVisualizer:
    """
    A class to visualize chiplet system mappings.
    """
    
    def __init__(self, adj_matrix_file=None, results_folder=None):
        """
        Initialize the ChipletVisualizer.
        
        Args:
            adj_matrix_file: Path to the CSV file containing the adjacency matrix
            results_folder: Path to the results folder containing network mappings
        """
        self.adj_matrix_file = adj_matrix_file
        self.results_folder = results_folder
        self.adj_matrix = None
        
        # Load adjacency matrix if provided
        if adj_matrix_file:
            self.adj_matrix = self.read_adjacency_matrix(adj_matrix_file)
    
    def read_adjacency_matrix(self, adj_matrix_file):
        """
        Read the adjacency matrix from a CSV file.
        
        Args:
            adj_matrix_file: Path to the CSV file containing the adjacency matrix
            
        Returns:
            numpy array containing the adjacency matrix
        """
        with open(adj_matrix_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            adj_matrix = np.array([list(map(int, filter(None, row))) for row in reader])
        
        return adj_matrix
    
    def read_network_mapping(self, mapping_file):
        """
        Read the network mapping from a JSON file.
        
        Args:
            mapping_file: Path to the JSON file containing the network mapping
            
        Returns:
            Dictionary containing the network mapping information
        """
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        return mapping_data
    
    def get_grid_dimensions(self, adj_matrix):
        """
        Calculate the dimensions of the grid based on the adjacency matrix.
        Assumes a mesh-like structure where each node has connections to neighbors.
        
        Args:
            adj_matrix: The adjacency matrix of the chiplet system
            
        Returns:
            Tuple (rows, cols) with the dimensions of the grid
        """
        n = adj_matrix.shape[0]
        
        # For a mesh, determine the dimensions by finding the maximum node with left/right connection
        # This is a simple approach for mesh topology
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                j = n // i
                if i <= j:  # Prefer more columns than rows
                    rows, cols = i, j
                else:
                    rows, cols = j, i
                    
        return rows, cols
    
    def create_chiplet_mapping_visualization(self, adj_matrix, mapping_data, output_dir):
        """
        Create a visualization of how the network is mapped to the chiplet system.
        
        Args:
            adj_matrix: The adjacency matrix of the chiplet system
            mapping_data: Dictionary containing the network mapping information or a MappedNetwork object
            output_dir: Directory to save the output image
        """
        # Check the type of mapping_data and extract network_name and model_idx
        if hasattr(mapping_data, 'model_name'):
            # It's a MappedNetwork object - use attribute access (dot notation)
            network_name = mapping_data.model_name
            
            # MappedNetwork objects now always have a unique model_idx attribute
            model_idx = str(mapping_data.model_idx)  # Ensure it's a string
            
            # For MappedNetwork objects, the mapping attribute contains the layer mapping
            if hasattr(mapping_data, 'mapping'):
                layer_mapping = mapping_data.mapping
            else:
                print(f"Warning: MappedNetwork object missing mapping attribute")
                layer_mapping = {}
            
        elif isinstance(mapping_data, dict) and "model_name" in mapping_data and "model_idx" in mapping_data:
            # It's a dictionary with the expected keys
            network_name = mapping_data["model_name"]
            model_idx = str(mapping_data["model_idx"])  # Ensure it's a string
            
            # For dictionaries, use dictionary access
            layer_mapping = mapping_data.get("layer_mapping", {})
        else:
            # If we can't determine network name or idx, use defaults and generate unique ID
            network_name = "unknown"
            model_idx = f"unknown_{int(time.time())}"
            print(f"Warning: Could not extract network name and index from mapping data: {type(mapping_data)}")
            layer_mapping = {}
        
        # Ensure network_name is valid for filenames
        network_name = network_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Get the dimensions of the grid
        n_nodes = adj_matrix.shape[0]
        rows, cols = self.get_grid_dimensions(adj_matrix)
        
        # Create a dictionary mapping chiplet IDs to layer mappings
        chiplet_to_layers = {}
        all_layer_ids = set()
        
        # Process the layer mapping - handle both list format for MappedNetwork.mapping and dict format for layer_mapping
        if isinstance(layer_mapping, list):
            # Format from MappedNetwork.mapping: [(layer_idx, [(chiplet_id, percentage), ...]), ...]
            for layer_idx, chiplet_list in layer_mapping:
                layer_id = str(layer_idx)  # Convert to string for consistency
                all_layer_ids.add(int(layer_id))
                
                for chiplet_id, percentage in chiplet_list:
                    if chiplet_id not in chiplet_to_layers:
                        chiplet_to_layers[chiplet_id] = []
                    
                    chiplet_to_layers[chiplet_id].append((int(layer_id), percentage))
        else:
            # Standard dictionary format: {"layer_id": [{"chiplet_id": id, "percentage": pct}, ...], ...}
            for layer_id, chiplets in layer_mapping.items():
                all_layer_ids.add(int(layer_id))
                for chiplet_info in chiplets:
                    chiplet_id = chiplet_info["chiplet_id"]
                    percentage = chiplet_info["percentage"]
                    
                    if chiplet_id not in chiplet_to_layers:
                        chiplet_to_layers[chiplet_id] = []
                    
                    chiplet_to_layers[chiplet_id].append((int(layer_id), percentage))
        
        # Create a fixed color map for all layer IDs
        all_layer_ids = sorted(list(all_layer_ids))
        # Use a color map that provides good distinction between colors
        cmap = plt.cm.get_cmap('tab20', max(20, len(all_layer_ids)))
        layer_colors = {layer_id: cmap(i % 20) for i, layer_id in enumerate(all_layer_ids)}
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw the chiplets as a grid
        for i in range(n_nodes):
            row = i // cols
            col = i % cols
            
            # Chiplet ID in the system is 1-indexed, so add 1 to the node index
            chiplet_id = i + 1
            
            # Center coordinates of the chiplet
            center_x = col + 0.45
            center_y = rows - row - 1 + 0.45
            
            # Draw the chiplet as a square
            square = plt.Rectangle((col, rows - row - 1), 0.9, 0.9, fill=False, edgecolor='black')
            ax.add_patch(square)
            
            # If this chiplet is used in the mapping, add layer information
            if chiplet_id in chiplet_to_layers:
                layers = chiplet_to_layers[chiplet_id]
                
                # Sort layers by percentage (higher percentage first)
                layers.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate total segments based on number of layers (at most 8 segments)
                num_layers = len(layers)
                total_segments = min(num_layers, 8)
                
                # If only one layer, just fill the square with a single color
                if num_layers == 1:
                    layer_id, percentage = layers[0]
                    color = layer_colors[layer_id]
                    
                    # Fill in the square
                    filled_square = plt.Rectangle(
                        (col, rows - row - 1), 0.9, 0.9, 
                        fill=True, 
                        facecolor=color,
                        alpha=0.6
                    )
                    ax.add_patch(filled_square)
                    
                    # Add layer number text
                    ax.text(center_x, center_y + 0.1, f"L{layer_id}", 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.1))
                else:
                    # For multiple layers, divide the square into sections
                    angle_per_segment = 360 / total_segments
                    start_angle = 90  # Start from the top
                    
                    # For exactly 2 layers, we need to handle the wedges differently to avoid mismatched labels
                    if total_segments == 2:
                        # Create explicit mapping to ensure correct layer ID with correct wedge segment
                        first_layer_id = layers[0][0]
                        second_layer_id = layers[1][0]
                        first_color = layer_colors[first_layer_id]
                        second_color = layer_colors[second_layer_id]
                        
                        # Draw first segment (left half)
                        wedge1 = mpatches.Wedge(
                            (center_x, center_y), 
                            0.44,
                            180, 0,  # Left half
                            fc=first_color,
                            ec='black',
                            alpha=0.6,
                            linewidth=0.5
                        )
                        ax.add_patch(wedge1)
                        
                        # Draw second segment (right half)
                        wedge2 = mpatches.Wedge(
                            (center_x, center_y), 
                            0.44,
                            0, 180,  # Right half
                            fc=second_color,
                            ec='black',
                            alpha=0.6,
                            linewidth=0.5
                        )
                        ax.add_patch(wedge2)
                        
                        # Position for first layer label - LEFT half
                        x_left = center_x - 0.25
                        circle1 = plt.Circle((x_left, center_y), 0.08, 
                                            facecolor=first_color, 
                                            edgecolor='white', 
                                            alpha=0.8,
                                            zorder=10)
                        ax.add_patch(circle1)
                        ax.text(x_left, center_y, f"{first_layer_id}", 
                                ha='center', va='center', fontsize=7, fontweight='bold',
                                color='white',
                                zorder=11)
                        
                        # Position for second layer label - RIGHT half
                        x_right = center_x + 0.25
                        circle2 = plt.Circle((x_right, center_y), 0.08, 
                                            facecolor=second_color, 
                                            edgecolor='white', 
                                            alpha=0.8,
                                            zorder=10)
                        ax.add_patch(circle2)
                        ax.text(x_right, center_y, f"{second_layer_id}", 
                                ha='center', va='center', fontsize=7, fontweight='bold',
                                color='white',
                                zorder=11)
                    else:
                        # Original code for 3+ segments
                        # Draw segments for each layer, up to total_segments
                        for idx, (layer_id, percentage) in enumerate(layers[:total_segments]):
                            end_angle = start_angle - angle_per_segment
                            
                            # Use consistent color from the predefined map
                            color = layer_colors[layer_id]
                            
                            # Draw segment
                            wedge = mpatches.Wedge(
                                (center_x, center_y), 
                                0.44,  # Slightly smaller than half the square
                                start_angle, 
                                end_angle, 
                                fc=color,
                                ec='black',  # Add edge color to separate segments
                                alpha=0.6,
                                linewidth=0.5
                            )
                            ax.add_patch(wedge)
                            
                            # Calculate midangle in degrees (clockwise from top)
                            midangle_degrees = (start_angle + end_angle) / 2
                            
                            # Convert to radians for trigonometric functions
                            midangle_radians = np.deg2rad(midangle_degrees)
                            
                            # Calculate text position - positioned more towards the outer edge of wedge
                            # for better visibility and alignment with color
                            radius = 0.30  # Slightly larger radius to position closer to edge
                            
                            # Calculate x,y position
                            x = center_x + radius * np.cos(midangle_radians)
                            y = center_y + radius * np.sin(midangle_radians)
                            
                            # Explicitly draw a background circle for the layer number
                            # that matches the wedge color for better identification
                            circle = plt.Circle((x, y), 0.08, 
                                                facecolor=color, 
                                                edgecolor='white', 
                                                alpha=0.8,
                                                zorder=10)
                            ax.add_patch(circle)
                            
                            # Add the text with no background (since we added a colored circle)
                            ax.text(x, y, f"{layer_id}", 
                                    ha='center', va='center', fontsize=7, fontweight='bold',
                                    color='white',
                                    zorder=11)
                            
                            # Update start angle for next segment
                            start_angle = end_angle
                    
                    # If there are more layers than segments, add a count
                    if num_layers > total_segments:
                        ax.text(center_x, center_y - 0.2, f"+{num_layers - total_segments} more", 
                                ha='center', va='center', fontsize=6,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.1))
            
            # Add chiplet ID text - make it more prominent
            # Position at bottom of square for better visibility
            ax.text(center_x, center_y - 0.25, 
                    str(chiplet_id), 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', 
                             alpha=0.9, pad=0.2),
                    color='white')
        
        # Draw edges between chiplets based on adjacency matrix
        for i in range(n_nodes):
            row_i = i // cols
            col_i = i % cols
            for j in range(i+1, n_nodes):  # Only process upper triangle of matrix
                if adj_matrix[i, j] == 1:  # If there is a connection
                    row_j = j // cols
                    col_j = j % cols
                    
                    # Draw a line between the chiplets
                    ax.plot([col_i + 0.45, col_j + 0.45], 
                            [rows - row_i - 1 + 0.45, rows - row_j - 1 + 0.45], 
                            'k-', alpha=0.3, linewidth=0.5)
        
        # Add legend for layer colors
        legend_elements = []
        for layer_id in all_layer_ids:
            legend_elements.append(
                mpatches.Patch(facecolor=layer_colors[layer_id], alpha=0.6, 
                               label=f'Layer {layer_id}')
            )
        
        # Add the legend if there are multiple layers
        if len(all_layer_ids) > 1:
            ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.1, 1), title="Layer Colors", 
                      fontsize=8, title_fontsize=9)
        
        # Set the limits and remove axis ticks
        ax.set_xlim(-0.1, cols + 0.5)
        ax.set_ylim(-0.1, rows + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        plt.title(f"Network {model_idx} ({network_name}) Mapping to Chiplet System")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save the figure as PNG with reduced complexity
            output_file_png = os.path.join(output_dir, f"network_{model_idx}_{network_name}_mapping.png")
            plt.savefig(output_file_png, dpi=150, bbox_inches='tight', format='png')
            print(f"Created PNG visualization: {output_file_png}")
        except Exception as e:
            print(f"Error saving PNG file: {e}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def visualize_all_network_mappings(self, adj_matrix_file=None, results_folder=None):
        """
        Visualize all network mappings in the specified results folder.
        
        Args:
            adj_matrix_file: Path to the CSV file containing the adjacency matrix
            results_folder: Path to the results folder containing network mappings
        """
        # Use provided values or fallback to instance variables
        adj_matrix_file = adj_matrix_file or self.adj_matrix_file
        results_folder = results_folder or self.results_folder
        
        if not adj_matrix_file or not results_folder:
            print("Error: Both adjacency matrix file and results folder must be provided.")
            return
        
        # Read the adjacency matrix if not already loaded
        if self.adj_matrix is None:
            self.adj_matrix = self.read_adjacency_matrix(adj_matrix_file)
        
        # Find the network mappings directory
        network_mappings_dir = os.path.join(results_folder, "network_mappings")
        
        if not os.path.exists(network_mappings_dir):
            print(f"Error: Network mappings directory not found: {network_mappings_dir}")
            return
        
        # Get the results folder name to use as subfolder name
        results_folder_name = os.path.basename(os.path.normpath(results_folder))
        
        # Create main visualization directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_vis_dir = os.path.join(script_dir, "network_mapping_visualizations")
        os.makedirs(main_vis_dir, exist_ok=True)
        
        # Create subfolder for this specific results folder
        output_dir = os.path.join(main_vis_dir, results_folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a static_mappings subdirectory for individual network mappings
        static_mappings_dir = os.path.join(output_dir, "static_mappings")
        os.makedirs(static_mappings_dir, exist_ok=True)
        
        # Find all network mapping JSON files
        mapping_files = [f for f in os.listdir(network_mappings_dir) 
                        if f.endswith("_mapping.json")]
        
        if not mapping_files:
            print(f"No network mapping files found in {network_mappings_dir}")
            return
        
        # Process each mapping file
        for mapping_file in mapping_files:
            mapping_path = os.path.join(network_mappings_dir, mapping_file)
            mapping_data = self.read_network_mapping(mapping_path)
            
            # Create visualization
            self.create_chiplet_mapping_visualization(self.adj_matrix, mapping_data, static_mappings_dir)
        
        return static_mappings_dir
    
    def visualize_network_mappings_from_data(self, retired_mapped_models, output_dir):
        """
        Visualize model mappings directly from provided model data.
        
        Args:
            retired_mapped_models: Dictionary of model mapping data where keys are model_idx
            output_dir: Directory to save the visualizations
        """
        if self.adj_matrix is None:
            print("Error: Adjacency matrix not loaded. Please provide it in the constructor or load it explicitly.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a 'static_mappings' subdirectory for individual network mappings
        static_mappings_dir = os.path.join(output_dir, "static_mappings")
        os.makedirs(static_mappings_dir, exist_ok=True)
        
        # Process each mapping - retired_mapped_models is a dictionary with model_idx as keys
        for model_idx, model_data in retired_mapped_models.items():      
            # Create visualization
            self.create_chiplet_mapping_visualization(self.adj_matrix, model_data, static_mappings_dir)
        
        return static_mappings_dir
    
    def visualize_system_state_over_time(self, retired_mapped_models, output_dir):
        """
        Visualize the system state at different time points, showing which models
        are active at each significant time point (model start or completion).
        
        Args:
            retired_mapped_models: Dictionary of model mapping data where keys are model_idx
            output_dir: Directory to save the visualizations
        """
        if self.adj_matrix is None:
            print("Error: Adjacency matrix not loaded. Please provide it in the constructor or load it explicitly.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a 'temporal_mappings' subdirectory for system state visualizations
        temporal_mappings_dir = os.path.join(output_dir, "temporal_mappings")
        os.makedirs(temporal_mappings_dir, exist_ok=True)
        
        # Collect all network start and completion times
        time_events = []
        for model_idx, model_data in retired_mapped_models.items():
            # Extract timing information based on whether it's a MappedNetwork object or a dictionary
            if hasattr(model_data, 'model_start_time_us') and hasattr(model_data, 'model_completion_time_us'):
                # It's a MappedNetwork object
                start_time = model_data.model_start_time_us
                completion_time = model_data.model_completion_time_us
                network_name = model_data.model_name
            elif isinstance(model_data, dict):
                # It's a dictionary
                start_time = model_data.get("model_start_time_us", -1)
                completion_time = model_data.get("model_completion_time_us", -1)
                network_name = model_data.get("model_name", f"unknown_{model_idx}")
            else:
                print(f"Warning: Unknown network data type for network {model_idx}")
                continue
            
            # Skip networks with invalid timing
            if start_time < 0 or completion_time < 0:
                print(f"Warning: Network {model_idx} ({network_name}) has invalid timing: start={start_time}, completion={completion_time}")
                continue
            
            # Add start and completion events
            time_events.append((start_time, "start", model_idx))
            time_events.append((completion_time, "completion", model_idx))
        
        # Sort events by time
        time_events.sort()
        
        # Find unique time points, avoiding duplicate time stamps
        unique_times = []
        prev_time = -1
        for event_time, event_type, _ in time_events:
            if abs(event_time - prev_time) > 0.001:  # Avoid floating point comparison issues
                unique_times.append(event_time)
                prev_time = event_time
        
        # For each unique time point, create a visualization of the system state
        for time_point in unique_times:
            # Find all networks active at this time
            active_networks = []
            for model_idx, model_data in retired_mapped_models.items():
                # Extract timing information based on object type
                if hasattr(model_data, 'model_start_time_us') and hasattr(model_data, 'model_completion_time_us'):
                    # It's a MappedNetwork object
                    start_time = model_data.model_start_time_us
                    completion_time = model_data.model_completion_time_us
                elif isinstance(model_data, dict):
                    # It's a dictionary
                    start_time = model_data.get("model_start_time_us", -1)
                    completion_time = model_data.get("model_completion_time_us", -1)
                else:
                    continue
                
                # Check if network is active at this time point
                if start_time <= time_point and (completion_time >= time_point or completion_time < 0):
                    active_networks.append((model_idx, model_data))
            
            # If there are active networks, create a visualization
            if active_networks:
                self._create_system_state_visualization(active_networks, time_point, temporal_mappings_dir)
        
        return temporal_mappings_dir
    
    def _create_system_state_visualization(self, active_networks, time_point, output_dir):
        """
        Create a visualization of the system state at a specific time point,
        showing multiple active networks with different colors.
        
        Args:
            active_networks: List of (model_idx, model_data) tuples representing active networks
            time_point: The time point (in microseconds) for this visualization
            output_dir: Directory to save the visualization
        """
        # Get the dimensions of the grid
        n_nodes = self.adj_matrix.shape[0]
        rows, cols = self.get_grid_dimensions(self.adj_matrix)
        
        # Create a mapping of chiplet IDs to the networks using them
        chiplet_to_networks = {}
        
        # Assign a unique color to each network
        network_colors = {}
        cmap = plt.cm.get_cmap('tab10', max(10, len(active_networks)))
        
        for i, (model_idx, model_data) in enumerate(active_networks):
            # Assign a color to this network
            network_colors[model_idx] = cmap(i % 10)
            
            # Extract the layer mapping based on whether it's a MappedNetwork object or dictionary
            if hasattr(model_data, 'mapping'):
                # It's a MappedNetwork object
                layer_mapping = model_data.mapping
                network_name = model_data.model_name
            elif isinstance(model_data, dict) and "layer_mapping" in model_data:
                # It's a dictionary with layer_mapping key
                layer_mapping = model_data["layer_mapping"]
                network_name = model_data.get("model_name", f"unknown_{model_idx}")
            else:
                print(f"Warning: Could not extract layer mapping for network {model_idx}")
                continue
            
            # Process the layer mapping - handle both list format for MappedNetwork.mapping and dict format for layer_mapping
            if isinstance(layer_mapping, list):
                # Format from MappedNetwork.mapping: [(layer_idx, [(chiplet_id, percentage), ...]), ...]
                for layer_idx, chiplet_list in layer_mapping:
                    for chiplet_id, percentage in chiplet_list:
                        if chiplet_id not in chiplet_to_networks:
                            chiplet_to_networks[chiplet_id] = []
                        chiplet_to_networks[chiplet_id].append((model_idx, network_name))
            else:
                # Standard dictionary format: {"layer_id": [{"chiplet_id": id, "percentage": pct}, ...], ...}
                for layer_id, chiplets in layer_mapping.items():
                    for chiplet_info in chiplets:
                        chiplet_id = chiplet_info["chiplet_id"]
                        if chiplet_id not in chiplet_to_networks:
                            chiplet_to_networks[chiplet_id] = []
                        chiplet_to_networks[chiplet_id].append((model_idx, network_name))
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw the chiplets as a grid
        for i in range(n_nodes):
            row = i // cols
            col = i % cols
            
            # Chiplet ID in the system is 1-indexed
            chiplet_id = i + 1
            
            # Center coordinates of the chiplet
            center_x = col + 0.45
            center_y = rows - row - 1 + 0.45
            
            # Draw the chiplet as a square
            square = plt.Rectangle((col, rows - row - 1), 0.9, 0.9, 
                                  fill=False, edgecolor='black')
            ax.add_patch(square)
            
            # If this chiplet is used by any networks, visualize it
            if chiplet_id in chiplet_to_networks:
                networks = chiplet_to_networks[chiplet_id]
                
                # If only one network is using this chiplet, fill it with that network's color
                if len(networks) == 1:
                    model_idx, _ = networks[0]
                    color = network_colors[model_idx]
                    
                    # Fill in the square
                    filled_square = plt.Rectangle(
                        (col, rows - row - 1), 0.9, 0.9, 
                        fill=True, 
                        facecolor=color,
                        alpha=0.6
                    )
                    ax.add_patch(filled_square)
                    
                    # Add network ID text
                    ax.text(center_x, center_y + 0.1, f"N{model_idx}", 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.1))
                else:
                    # For multiple networks, divide the square into sections
                    num_networks = min(len(networks), 8)  # Limit to 8 segments for readability
                    angle_per_segment = 360 / num_networks
                    start_angle = 90  # Start from the top
                    
                    # Handle special case for 2 networks
                    if num_networks == 2:
                        first_model_idx, _ = networks[0]
                        second_model_idx, _ = networks[1]
                        first_color = network_colors[first_model_idx]
                        second_color = network_colors[second_model_idx]
                        
                        # Draw first segment (left half)
                        wedge1 = mpatches.Wedge(
                            (center_x, center_y), 
                            0.44,
                            180, 0,  # Left half
                            fc=first_color,
                            ec='black',
                            alpha=0.6,
                            linewidth=0.5
                        )
                        ax.add_patch(wedge1)
                        
                        # Draw second segment (right half)
                        wedge2 = mpatches.Wedge(
                            (center_x, center_y), 
                            0.44,
                            0, 180,  # Right half
                            fc=second_color,
                            ec='black',
                            alpha=0.6,
                            linewidth=0.5
                        )
                        ax.add_patch(wedge2)
                        
                        # Position for first network label - LEFT half
                        x_left = center_x - 0.25
                        circle1 = plt.Circle((x_left, center_y), 0.08, 
                                            facecolor=first_color, 
                                            edgecolor='white', 
                                            alpha=0.8,
                                            zorder=10)
                        ax.add_patch(circle1)
                        ax.text(x_left, center_y, f"N{first_model_idx}", 
                                ha='center', va='center', fontsize=7, fontweight='bold',
                                color='white',
                                zorder=11)
                        
                        # Position for second network label - RIGHT half
                        x_right = center_x + 0.25
                        circle2 = plt.Circle((x_right, center_y), 0.08, 
                                            facecolor=second_color, 
                                            edgecolor='white', 
                                            alpha=0.8,
                                            zorder=10)
                        ax.add_patch(circle2)
                        ax.text(x_right, center_y, f"N{second_model_idx}", 
                                ha='center', va='center', fontsize=7, fontweight='bold',
                                color='white',
                                zorder=11)
                    else:
                        # Handle 3+ networks with wedge segments
                        for idx, (model_idx, _) in enumerate(networks[:num_networks]):
                            end_angle = start_angle - angle_per_segment
                            
                            # Use this network's color
                            color = network_colors[model_idx]
                            
                            # Draw segment
                            wedge = mpatches.Wedge(
                                (center_x, center_y), 
                                0.44,
                                start_angle, 
                                end_angle, 
                                fc=color,
                                ec='black',
                                alpha=0.6,
                                linewidth=0.5
                            )
                            ax.add_patch(wedge)
                            
                            # Calculate position for network ID
                            midangle_degrees = (start_angle + end_angle) / 2
                            midangle_radians = np.deg2rad(midangle_degrees)
                            radius = 0.30
                            
                            x = center_x + radius * np.cos(midangle_radians)
                            y = center_y + radius * np.sin(midangle_radians)
                            
                            # Add network ID label with colored background
                            circle = plt.Circle((x, y), 0.08, 
                                                facecolor=color, 
                                                edgecolor='white', 
                                                alpha=0.8,
                                                zorder=10)
                            ax.add_patch(circle)
                            
                            ax.text(x, y, f"N{model_idx}", 
                                    ha='center', va='center', fontsize=7, fontweight='bold',
                                    color='white',
                                    zorder=11)
                            
                            # Update start angle for next segment
                            start_angle = end_angle
                    
                    # If there are more networks than segments, add a count
                    if len(networks) > num_networks:
                        ax.text(center_x, center_y - 0.2, f"+{len(networks) - num_networks} more", 
                                ha='center', va='center', fontsize=6,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.1))
            
            # Add chiplet ID text - make it more prominent
            ax.text(center_x, center_y - 0.25, 
                    str(chiplet_id), 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', edgecolor='white', 
                             alpha=0.9, pad=0.2),
                    color='white')
        
        # Draw edges between chiplets based on adjacency matrix
        for i in range(n_nodes):
            row_i = i // cols
            col_i = i % cols
            for j in range(i+1, n_nodes):  # Only process upper triangle of matrix
                if self.adj_matrix[i, j] == 1:  # If there is a connection
                    row_j = j // cols
                    col_j = j % cols
                    
                    # Draw a line between the chiplets
                    ax.plot([col_i + 0.45, col_j + 0.45], 
                            [rows - row_i - 1 + 0.45, rows - row_j - 1 + 0.45], 
                            'k-', alpha=0.3, linewidth=0.5)
        
        # Add legend for network colors
        legend_elements = []
        for model_idx, model_data in active_networks:
            if hasattr(model_data, 'model_name'):
                display_name = model_data.model_name
            elif isinstance(model_data, dict) and "model_name" in model_data:
                display_name = model_data["model_name"]
            else:
                display_name = f"Network {model_idx}"
            
            # Shorten very long network names
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."
            
            legend_elements.append(
                mpatches.Patch(facecolor=network_colors[model_idx], alpha=0.6, 
                               label=f'N{model_idx}: {display_name}')
            )
        
        # Add the legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.1, 1), title="Active Networks", 
                      fontsize=8, title_fontsize=9)
        
        # Set the limits and remove axis ticks
        ax.set_xlim(-0.1, cols + 0.5)
        ax.set_ylim(-0.1, rows + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title with timestamp
        plt.title(f"System State at {time_point:.2f} μs - {len(active_networks)} Active Networks")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format time for filename (with microsecond precision)
        formatted_time = f"{time_point:.2f}".replace('.', '_')
        
        try:
            # Save the figure as PNG
            output_file_png = os.path.join(output_dir, f"system_state_t{formatted_time}us.png")
            plt.savefig(output_file_png, dpi=150, bbox_inches='tight', format='png')
            print(f"Created PNG system state visualization at {time_point:.2f} μs: {output_file_png}")
        except Exception as e:
            print(f"Error saving PNG file: {e}")
        
        # Close the figure to free memory
        plt.close(fig)
