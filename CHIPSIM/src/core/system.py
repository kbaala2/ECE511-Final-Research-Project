import networkx as nx
import numpy as np
import yaml

from assets.chiplet_specs.chiplet_params import CHIPLET_TYPES
from .chiplet import Chiplet

# load material properties from yaml file
def load_dict_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        dict = yaml.safe_load(f)
    return dict

class System:
    """
    Models and manages a cluster of chiplets, tracking crossbar availability.
    """

    def __init__(
        self,
        chiplet_mapping_file,
        adj_matrix_file,
    ):
        """
        Initializes the SystemState with chiplet configurations and simulation parameters.

        Parameters:
            chiplet_mapping_file (str): YAML file containing chiplet mapping.
            adj_matrix_file (str): File containing the adjacency matrix for chiplet connections.
        """
        
        # Load chiplet mapping from YAML
        self.chiplet_mapping = load_dict_yaml(chiplet_mapping_file)
        self.num_chiplets = len(self.chiplet_mapping)
        self.chiplets = [
            Chiplet(chiplet_id=i + 1, chiplet_type=self.chiplet_mapping[i + 1])
            for i in range(self.num_chiplets)
        ]
        
        # Initialize the total memory in the system
        self.system_memory = 0
        
        # Get the total amount of memory in the system, in bits
        for chiplet in self.chiplets:
            self.system_memory += chiplet.get_total_memory()

        # Capacity model is used for availability; per-crossbar memory may still be used via chiplet API when needed
        
        # Load adjacency matrix
        self.adj_matrix = np.loadtxt(adj_matrix_file)
        
        # Build the chiplet network from the adjacency matrix
        self.chiplet_network = self.build_chiplet_network()
        
        # Identify I/O and compute chiplets
        self.io_chiplet_ids = []
        self.compute_chiplet_ids = []
        self._identify_chiplet_types()
        
    def build_chiplet_network(self):
        """
        Build the chiplet network topology from the adjacency matrix file and chiplet mapping,
        including energy efficiency and performance metrics for each chiplet.

        Args:
            adj_matrix_file (str): Filename of the adjacency matrix.
            chiplet_mapping (dict): Mapping from chiplet numbers to chiplet types.

        Returns:
            networkx.Graph: Graph representing the chiplet network with additional metrics.
        """
        # Convert CHIPLET_TYPES list to a dictionary for efficient lookup
        CHIPLET_TYPES_DICT = {chiplet['name']: chiplet for chiplet in CHIPLET_TYPES}

        # Create a graph
        G = nx.Graph()

        # Add nodes with chiplet types and additional metrics
        for chiplet_num in range(self.num_chiplets):
            chiplet_id = chiplet_num + 1  # Chiplet numbers start from 1
            chiplet_type = self.chiplet_mapping.get(chiplet_id, 'Unknown')

            # Retrieve metrics from CHIPLET_TYPES_DICT
            chiplet_info = CHIPLET_TYPES_DICT.get(chiplet_type, {})
            compute_type = chiplet_info.get('type', 'Unknown')  # IMC, CMOS, or IO
            energy_efficiency = chiplet_info.get('energy_per_mac', 0)  # Default to 0 if not found

            # Add node with all attributes
            G.add_node(
                chiplet_id,
                chiplet_type=chiplet_type,
                compute_type=compute_type,
                energy_efficiency=energy_efficiency,
            )
        
        total_nodes = self.num_chiplets

        # Add edges based on adjacency matrix
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                if self.adj_matrix[i, j] == 1:
                    G.add_edge(i + 1, j + 1)  # Adjust index to start from 1

        return G
    
    def _identify_chiplet_types(self):
        """Identify and categorize I/O vs compute chiplets based on their type"""
        for chiplet in self.chiplets:
            if chiplet.type == "IO":
                self.io_chiplet_ids.append(chiplet.id)
            else:
                self.compute_chiplet_ids.append(chiplet.id)
    
    def is_io_chiplet(self, chiplet_id: int) -> bool:
        """Check if a chiplet is an I/O chiplet"""
        return chiplet_id in self.io_chiplet_ids
    
    def is_compute_chiplet(self, chiplet_id: int) -> bool:
        """Check if a chiplet is a compute chiplet"""
        return chiplet_id in self.compute_chiplet_ids
    
    def get_chiplet_compute_type(self, chiplet_id: int) -> str:
        """
        Get the compute type (IMC, CMOS, or IO) for a specific chiplet.
        
        Args:
            chiplet_id: ID of the chiplet
            
        Returns:
            str: Compute type ('IMC', 'CMOS', or 'IO')
        """
        if chiplet_id not in self.chiplet_network:
            raise ValueError(f"Chiplet ID {chiplet_id} not found in network")
        return self.chiplet_network.nodes[chiplet_id].get('compute_type', 'Unknown')
    
    def get_chiplet_params(self, chiplet_id: int) -> dict:
        """
        Get all parameters for a specific chiplet from CHIPLET_TYPES.
        
        Args:
            chiplet_id: ID of the chiplet
            
        Returns:
            dict: Chiplet parameters including type, energy_per_mac, etc.
        """
        from assets.chiplet_specs.chiplet_params import CHIPLET_TYPES
        
        chiplet_type_name = self.chiplet_mapping.get(chiplet_id, 'Unknown')
        for chiplet_params in CHIPLET_TYPES:
            if chiplet_params['name'] == chiplet_type_name:
                return chiplet_params
        return {}
    
    def get_edge_chiplets(self) -> list:
        """Get compute chiplets at the edge of the mesh (connected to I/O)"""
        edge_chiplets = []
        for io_id in self.io_chiplet_ids:
            # Find all compute chiplets connected to this I/O chiplet
            if io_id in self.chiplet_network:
                neighbors = list(self.chiplet_network.neighbors(io_id))
                edge_chiplets.extend([n for n in neighbors if self.is_compute_chiplet(n)])
        return list(set(edge_chiplets))
    
    def get_io_chiplet_for_compute(self, compute_chiplet_id: int) -> int:
        """
        Determine which I/O chiplet should serve a compute chiplet
        based on network distance (shortest path).
        
        Args:
            compute_chiplet_id: ID of the compute chiplet
            
        Returns:
            ID of the closest I/O chiplet, or None if no I/O chiplets exist
        """
        if not self.io_chiplet_ids:
            return None
            
        min_distance = float('inf')
        closest_io = None
        
        for io_id in self.io_chiplet_ids:
            distance = self.get_path_length_between_chiplets(io_id, compute_chiplet_id)
            if distance != -1 and distance < min_distance:
                min_distance = distance
                closest_io = io_id
                
        return closest_io
    
    def get_io_chiplet_connections(self) -> dict:
        """
        Get mapping of I/O chiplets to their directly connected compute chiplets.
        
        Returns:
            Dictionary mapping I/O chiplet IDs to lists of connected compute chiplet IDs
        """
        connections = {}
        for io_id in self.io_chiplet_ids:
            if io_id in self.chiplet_network:
                connections[io_id] = [
                    n for n in self.chiplet_network.neighbors(io_id) 
                    if self.is_compute_chiplet(n)
                ]
            else:
                connections[io_id] = []
        return connections

    def get_available_memory_in_system(self):
        """
        Retrieves the current number of available memory in int8 units in each chiplet.

        Returns:
            int: Total available memory in the system.
        """
        available_memory = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            available_memory[chiplet_id - 1] = chiplet.get_available_memory()
        
        return available_memory

    # Crossbar-specific availability API removed in favor of capacity-based API

    def get_available_capacity_per_chiplet(self):
        """
        Retrieves the current number of available capacity units in each chiplet.
        IMC: capacity unit == crossbar; CMOS: capacity unit == single weight.

        Returns:
            numpy.ndarray: Array of available capacity units per chiplet.
        """
        capacity = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            capacity[chiplet_id - 1] = chiplet.get_available_capacity_units()
        return capacity
    
    # memory_per_crossbar system-level array removed; query chiplet.memory_per_crossbar when needed

    def update_capacity_availability(self, remaining_capacity_units):
        """
        Updates the available capacity units in each chiplet based on mapping results.
        IMC: updates available crossbars; CMOS: updates available weights.

        Args:
            remaining_capacity_units (np.array): Remaining capacity units per chiplet.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        if len(remaining_capacity_units) != self.num_chiplets:
            print(f"System Class Error: Expected {self.num_chiplets} values in remaining_capacity_units array, got {len(remaining_capacity_units)}")
            exit(1)

        for i, chiplet in enumerate(self.chiplets):
            if not chiplet.set_available_capacity_units(int(remaining_capacity_units[i])):
                return False
        return True
    
    def get_total_system_memory(self):
        """
        Get the total memory capacity of the entire system (all chiplets).
        
        Returns:
            float: Total system memory in weight units
        """
        return self.system_memory
    
    def get_model_memory_requirement(self, model_metrics):
        """
        Calculate the total memory requirement for a model.
        
        Args:
            model_metrics (list): A list of dictionaries, where each dict describes a layer.
            
        Returns:
            float: Total memory requirement in weight units
        """
        # Use crossbars_required for memory requirement calculation
        return sum(layer['crossbars_required'] for layer in model_metrics)
    
    def can_model_ever_fit(self, model_metrics):
        """
        Check if a model can ever fit in the system by comparing against total system memory.
        This is a definitive check - if this returns False, the model will never fit.
        
        Args:
            model_metrics (list): A list of dictionaries, where each dict describes a layer.
            
        Returns:
            tuple: (bool, str or None). True if model can potentially fit,
                   False with detailed error message if it can never fit.
        """
        total_required_weights = self.get_model_memory_requirement(model_metrics)
        total_system_memory = self.get_total_system_memory()
        
        if total_required_weights > total_system_memory:
            return False, f"MODEL_TOO_LARGE_FOR_SYSTEM"
        
        return True, None

    def can_model_fit_now(self, model_metrics):
        """
        Check if a model can fit given current available memory across chiplets.
        Uses per-chiplet available memory (supports both IMC and CMOS semantics).

        Args:
            model_metrics (list): Layer metrics for the model

        Returns:
            tuple: (bool, str or None)
        """
        total_required_weights = self.get_model_memory_requirement(model_metrics)
        total_available_memory = int(self.get_available_memory_in_system().sum())

        if total_required_weights > total_available_memory:
            return False, "INSUFFICIENT_MEMORY"
        return True, None

    # Crossbar-specific update API removed; use update_capacity_availability instead
        
    def get_path_length_between_chiplets(self, chiplet_id1, chiplet_id2):
        """
        Calculate the number of links that need to be traversed between two chiplets
        in the chiplet network (shortest path).
        
        Args:
            chiplet_id1 (int): ID of the first chiplet
            chiplet_id2 (int): ID of the second chiplet
            
        Returns:
            int: Number of links between the chiplets, or -1 if no path exists
        """
        # Check if chiplet IDs are valid
        if chiplet_id1 < 1 or chiplet_id1 > self.num_chiplets or chiplet_id2 < 1 or chiplet_id2 > self.num_chiplets:
            print(f"Error: Invalid chiplet ID. Valid IDs are 1-{self.num_chiplets}")
            return -1
            
        # If the chiplets are the same, no links need to be traversed
        if chiplet_id1 == chiplet_id2:
            return 0
            
        try:
            # Use NetworkX to calculate the shortest path length
            path_length = nx.shortest_path_length(self.chiplet_network, source=chiplet_id1, target=chiplet_id2)
            return path_length
        except nx.NetworkXNoPath:
            # No path exists between the chiplets
            print(f"No path exists between chiplet {chiplet_id1} and chiplet {chiplet_id2}")
            return -1
        except Exception as e:
            print(f"Error calculating path length: {str(e)}")
            return -1