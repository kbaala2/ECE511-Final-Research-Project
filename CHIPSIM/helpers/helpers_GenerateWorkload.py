#!/usr/bin/env python3

import os
import sys
import csv
import random
import math
import numpy as np
import warnings

# Add the DNN_models directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "assets", "DNN_models"))
try:
    import model_definitions
except ImportError:
    print("Error: Could not import model definitions. Make sure the path is correct.")
    sys.exit(1)

def generate_workload(network_config, total_runtime_us, injection_rate, input_range=(1, 10), output_file=None):
    """
    Generate a workload file for the chiplet system simulator.
    
    Args:
        network_config (dict): Dictionary defining networks and their selection probability.
                               Format: {'network_name': {'probability': float}, ...}
                               Probabilities will be normalized if they don't sum to 1.
        total_runtime_us (int): Total runtime of the simulation in microseconds
        injection_rate (float): Rate of network injection per time step (1us)
        input_range (tuple): Range of the number of inputs to apply (min, max)
        output_file (str): Path to the output CSV file. If None, a default name will be used.
        
    Returns:
        str: Path to the generated workload file
    """
    # Validate network config
    valid_networks = []
    probabilities = []
    
    total_prob = 0
    for network_name, config in network_config.items():
        if 'probability' not in config or not isinstance(config['probability'], (int, float)) or config['probability'] < 0:
            raise ValueError(f"Invalid or missing 'probability' for network '{network_name}'.")
        
        if network_name in model_definitions.NETWORK_DEFINITIONS:
            valid_networks.append(network_name)
            probabilities.append(config['probability'])
            total_prob += config['probability']
        else:
            warnings.warn(f"Network '{network_name}' not found in model_definitions. Skipping.", UserWarning)

    if not valid_networks:
        raise ValueError("No valid networks specified in network_config.")

    # Normalize probabilities
    if not math.isclose(total_prob, 1.0):
        warnings.warn(f"Probabilities do not sum to 1 (sum={total_prob}). Normalizing.", UserWarning)
        if total_prob == 0:
             # Assign equal probability if sum is 0
             probabilities = [1.0 / len(valid_networks)] * len(valid_networks)
        else:
             probabilities = [p / total_prob for p in probabilities]
    
    # Set default output file name if not provided
    if output_file is None:
        output_dir = os.path.join(os.getcwd(), "workloads")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"workload_{injection_rate}rate_{total_runtime_us}us_{input_range[0]}-{input_range[1]}.csv")
    
    # Generate random injection times over the total runtime
    # Using a Poisson distribution to model network arrivals with the given injection rate
    
    # Calculate expected number of networks based on injection rate and runtime
    expected_num_networks = int(injection_rate * total_runtime_us)
    
    if expected_num_networks <= 1:
        injection_times = [0]  # Just one network at time 0
        num_networks = 1
    else:
        # Generate Poisson-distributed intervals using the given injection rate
        intervals = np.random.exponential(1/injection_rate, expected_num_networks * 2)  # Generate more than needed
        
        # Convert intervals to absolute times and filter those within total runtime
        cumulative_times = np.cumsum(intervals)
        injection_times = [math.floor(t) for t in cumulative_times if t < total_runtime_us]
        
        # If we ended up with no networks, add at least one
        if not injection_times:
            injection_times = [0]
        
        # Sort the injection times
        injection_times.sort()
        num_networks = len(injection_times)
    
    # Generate the workload data
    workload_data = []
    for i in range(num_networks):
        # Randomly select a network based on probabilities
        network = random.choices(valid_networks, weights=probabilities, k=1)[0]
        # Randomly select number of inputs
        num_inputs = random.randint(input_range[0], input_range[1])
        # Add to workload (adding net_idx which is 1-indexed)
        workload_data.append({
            'net_idx': i + 1,
            'inject_time_us': injection_times[i],
            'network': network,
            'num_inputs': num_inputs
        })
    
    # Sort by injection time
    workload_data.sort(key=lambda x: x['inject_time_us'])
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as file:
        fieldnames = ['net_idx', 'inject_time_us', 'network', 'num_inputs']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in workload_data:
            writer.writerow(item)
    
    print(f"Workload file generated: {output_file}")
    print(f"Contains {num_networks} networks spanning {total_runtime_us}us")
    print(f"Injection rate: {injection_rate} networks per us")
    
    return output_file

def create_sample_workload():
    """
    Create a sample workload with fixed parameters and network probabilities.
    Modify these parameters as needed.
    """
    # Define networks and their selection probabilities
    # Ensure these network names exist in model_definitions.py
    network_config = {
        "alexnet":  {'probability': 0.25},  # 50% chance
        #"vgg16":    {'probability': 0.1},   # 20% chance
        "resnet18": {'probability': 0.25},  # 30% chance
        "resnet34": {'probability': 0.25},    # 40% chance
        "resnet50": {'probability': 0.25},    # 40% chance
        #"mobilenet_v3": {'probability': 0.1},    # 40% chance
    }
    
    # Set workload parameters
    total_runtime_us = 50  # Total runtime in microseconds
    injection_rate = 1        # Measured in networks per us
    input_range = (7, 7)       # Set the range of number of inputs per network
    
    # Output file (leave as None for default naming, or specify a path)
    output_file = None
    
    # Generate the workload
    return generate_workload(
        network_config=network_config,
        total_runtime_us=total_runtime_us,
        injection_rate=injection_rate,
        input_range=input_range,
        output_file=output_file
    )

if __name__ == "__main__":
    create_sample_workload()