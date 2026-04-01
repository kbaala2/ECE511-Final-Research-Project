import numpy as np
import csv
import os

def generate_mesh_adj_matrix(x_dim, y_dim, output_filename=None):
    """
    Generate an adjacency matrix for a mesh network.
    
    Parameters:
    -----------
    x_dim : int
        Number of nodes in the x direction
    y_dim : int
        Number of nodes in the y direction
    output_filename : str, optional
        Name of the output file. If None, no file is saved.
        
    Returns:
    --------
    numpy.ndarray
        The adjacency matrix
    """
    total_nodes = x_dim * y_dim
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    
    # Generate connections
    for i in range(total_nodes):
        # Convert 1D index to 2D coordinates
        row = i // x_dim
        col = i % x_dim
        
        # Connect to top neighbor
        if row > 0:
            top_neighbor = (row - 1) * x_dim + col
            adj_matrix[i, top_neighbor] = 1
            adj_matrix[top_neighbor, i] = 1
            
        # Connect to bottom neighbor
        if row < y_dim - 1:
            bottom_neighbor = (row + 1) * x_dim + col
            adj_matrix[i, bottom_neighbor] = 1
            adj_matrix[bottom_neighbor, i] = 1
            
        # Connect to left neighbor
        if col > 0:
            left_neighbor = row * x_dim + (col - 1)
            adj_matrix[i, left_neighbor] = 1
            adj_matrix[left_neighbor, i] = 1
            
        # Connect to right neighbor
        if col < x_dim - 1:
            right_neighbor = row * x_dim + (col + 1)
            adj_matrix[i, right_neighbor] = 1
            adj_matrix[right_neighbor, i] = 1
    
    # Save to file if filename is provided
    if output_filename:
        directory = os.path.dirname(output_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for row in adj_matrix:
                writer.writerow(row)
    
    return adj_matrix

def main():
    """
    Example usage of the adjacency matrix generator.
    """
    # Create a 10x10 mesh network
    x_dim = 10
    y_dim = 10
    output_file = "system_topologies/adj_matrix_10x10_mesh.csv"
    
    adj_matrix = generate_mesh_adj_matrix(x_dim, y_dim, output_file)
    print(f"Created a {x_dim}x{y_dim} mesh network adjacency matrix")
    print(f"Total nodes: {x_dim * y_dim}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
