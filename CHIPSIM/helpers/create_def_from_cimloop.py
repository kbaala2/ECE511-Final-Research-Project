import os
import yaml
import json
from pathlib import Path
import re
import pprint
from collections import OrderedDict

# Dictionary to map letter variables to readable names
VARIABLE_MAPPING = {
    'N': 'batch_size',
    'X': 'input_precision',
    'C': 'input_channels',
    'H': 'input_height',
    'W': 'input_width',
    'G': 'groups',
    'Y': 'weight_precision',
    'R': 'weight_height',
    'S': 'weight_width',
    'Hdilation': 'height_dilation',
    'Hstride': 'height_stride',
    'Wdilation': 'width_dilation',
    'Wstride': 'width_stride',
    'Z': 'output_precision',
    'M': 'output_channels',
    'P': 'output_height',
    'Q': 'output_width'
}

# Hardcoded list of network folders to include
NETWORK_FOLDERS = [
    'alexnet',
    'densenet201',
    'dpt_large',
    'vision_transformer',
    'vision_transformer_qkv_fusion',
    'gpt2_medium',
    'mobilebert',
    'mobilenet_v3',
    'msft_phi_1_5',
    'resnet18',
    'vgg16',
    'resnet34',
    'resnet50',
    'mobilenet_v2'
]

# Default value substitutions (for named defaults in problem_base)
DEFAULT_SUBSTITUTIONS = {
    'BATCH_SIZE': 1,
    'ENCODED_INPUT_BITS': 8,
    'ENCODED_WEIGHT_BITS': 8,
    'ENCODED_OUTPUT_BITS': 32
}

# Communication rules per network
# Expandable: add entries for other networks as needed
def compute_custom_receiving_layers(network_name, idx, sorted_keys, temp_layers):
    """Return a custom receiving list for the given layer index or None to use default.

    For 'vision_transformer_qkv_fusion':
      - If the current layer is the fused QKV (Linear with output_channels=2304 and output_height=197),
        connect it to the next two layers that are ViTSelfAttention layers (when available).
    """
    try:
        if network_name == 'vision_transformer_qkv_fusion':
            # Current layer dict
            cur_key = sorted_keys[idx]
            cur_layer = temp_layers[cur_key]
            params = cur_layer.get('parameters', {})
            desc = cur_layer.get('description', '')

            # Identify fused QKV by structure
            is_linear = desc.lower().startswith('linear')
            m = params.get('output_channels')
            p = params.get('output_height')
            if is_linear and m == 2304 and p == 197:
                targets = []
                # candidates are the next two layers, if they exist and are ViTSelfAttention
                for off in (1, 2):
                    if idx + off < len(sorted_keys):
                        k = sorted_keys[idx + off]
                        nxt = temp_layers[k]
                        if nxt.get('description', '').lower().startswith('vitselfattention'):
                            targets.append(k if isinstance(k, int) else int(k) if isinstance(k, str) and k.isdigit() else k)
                if targets:
                    return targets
        return None
    except Exception:
        return None

def process_yaml_include(yaml_content, base_dir):
    """Process YAML include directives."""
    lines = yaml_content.split('\n')
    processed_lines = []
    
    for line in lines:
        if '{{include_text(' in line:
            include_path = line.split("'")[1] if "'" in line else line.split('"')[1]
            # Handle relative paths
            if include_path.startswith('../'):
                include_path = os.path.normpath(os.path.join(base_dir, include_path))
            elif not os.path.isabs(include_path):
                include_path = os.path.join(base_dir, include_path)
                
            with open(include_path, 'r') as include_file:
                processed_lines.extend(include_file.read().split('\n'))
        else:
            processed_lines.append(line)
            
    return '\n'.join(processed_lines)

def get_default_values(networks_dir):
    """Extract default values from problem_base.yaml"""
    problem_base_path = os.path.join(networks_dir, 'problem_base.yaml')
    
    with open(problem_base_path, 'r') as f:
        content = f.read()
    
    data = yaml.safe_load(content)
    
    # Extract the default values from problem_base
    default_values = {}
    if 'problem_base_ignore' in data and 'instance' in data['problem_base_ignore']:
        instance = data['problem_base_ignore']['instance']
        
        # Process each default value
        for key, value in instance.items():
            if isinstance(value, str) and value in DEFAULT_SUBSTITUTIONS:
                # Substitute named defaults with their values
                default_values[key] = DEFAULT_SUBSTITUTIONS[value]
            else:
                default_values[key] = value
    
    return default_values

def convert_to_readable_dict(yaml_file, default_values):
    """Convert a YAML file to a dictionary with readable names."""
    base_dir = os.path.dirname(yaml_file)
    
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    # Process includes
    processed_content = process_yaml_include(yaml_content, base_dir)
    
    # Parse YAML
    data = yaml.safe_load(processed_content)
    
    # Extract problem instance
    if 'problem' in data and 'instance' in data['problem']:
        instance = data['problem']['instance']
    else:
        # Fallback if structure is different
        instance = {}
        for key in data:
            if isinstance(data[key], dict) and 'instance' in data[key]:
                instance = data[key]['instance']
                break
    
    # Merge with default values
    # Start with all defaults
    merged_instance = default_values.copy()
    # Override with values from the instance
    merged_instance.update(instance)
    
    # Convert to readable dictionary
    readable_dict = {}
    
    # Skip storing the name field
    layer_name = data.get('problem', {}).get('name', 'Unknown')
    
    # Convert parameters
    readable_dict['parameters'] = {}
    for key, value in merged_instance.items():
        if key in VARIABLE_MAPPING:
            readable_dict['parameters'][VARIABLE_MAPPING[key]] = value
        else:
            readable_dict['parameters'][key] = value
    
    # Generate a description based on available data
    if 'problem' in data and 'name' in data['problem']:
        description = f"{data['problem']['name']} layer"
    else:
        description = "Unknown layer type"
    
    readable_dict['description'] = description
    
    return readable_dict

def extract_layer_number(filename):
    """Extract the layer number from a filename like 000.yaml, 01.yaml, etc."""
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Try to convert to integer (this handles leading zeros)
    try:
        layer_num = int(name)
        return layer_num
    except ValueError:
        # If conversion fails, return the original name
        return name

def main():
    # Set path to networks directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    networks_dir = os.path.normpath(os.path.join(script_dir, '..', 'assets', 'DNN_models'))
    
    # Get default values from problem_base.yaml
    default_values = get_default_values(networks_dir)
    print(f"Loaded default values: {default_values}")
    
    # Dictionary to store the network data
    network_dict = {}
    
    # Process each network folder from our hardcoded list
    for network_name in NETWORK_FOLDERS:
        network_path = os.path.join(networks_dir, network_name)
        
        # Skip if the folder doesn't exist
        if not os.path.exists(network_path):
            print(f"Warning: Network folder {network_name} not found, skipping.")
            continue
        
        # Prepare entry for this network
        network_dict[network_name] = {}
        
        # Get all yaml files in this network folder (numeric filenames only)
        yaml_files = []
        for file in os.listdir(network_path):
            if file.endswith('.yaml') and file != 'problem_base.yaml':
                stem = os.path.splitext(file)[0]
                if stem.isdigit():
                    yaml_files.append(os.path.join(network_path, file))
        
        # Sort files by layer number (numerical order, not lexicographical)
        yaml_files.sort(key=lambda file: extract_layer_number(os.path.basename(file)))
        
        # Process each layer file and store in a temporary dictionary with numeric keys
        temp_layers = {}
        for yaml_file in yaml_files:
            filename = os.path.basename(yaml_file)
            layer_num = extract_layer_number(filename)
            
            # Convert file to readable dictionary
            layer_dict = convert_to_readable_dict(yaml_file, default_values)
            
            # Add layer to temporary dictionary with numeric key
            temp_layers[layer_num] = layer_dict
        
        # Sort the layers by numeric index and add to network dictionary under 'layers'
        # Ensure numeric sorting by explicitly converting keys to integers when possible
        sorted_keys = sorted(temp_layers.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x)

        # Build the 'layers' mapping with integer keys and receiving_layers
        layers_mapping = OrderedDict()
        for idx, layer_key in enumerate(sorted_keys):
            # Determine the numeric key for the layer
            if isinstance(layer_key, int):
                numeric_key = layer_key
            elif isinstance(layer_key, str) and layer_key.isdigit():
                numeric_key = int(layer_key)
            else:
                # Skip non-numeric layers as receiving_layers cannot be determined reliably
                continue

            # Determine receiving layers, with custom rules hook
            custom = compute_custom_receiving_layers(network_name, idx, sorted_keys, temp_layers)
            if custom is not None and len(custom) > 0:
                receiving = custom
            else:
                # default: next layer or -1 for last
                if idx < len(sorted_keys) - 1:
                    next_key = sorted_keys[idx + 1]
                    if isinstance(next_key, int):
                        receiving = [next_key]
                    elif isinstance(next_key, str) and next_key.isdigit():
                        receiving = [int(next_key)]
                    else:
                        receiving = [-1]
                else:
                    receiving = [-1]

            # Copy and augment the layer dictionary
            layer_entry = temp_layers[layer_key]
            layer_entry = dict(layer_entry)
            layer_entry['receiving_layers'] = receiving

            layers_mapping[numeric_key] = layer_entry

        network_dict[network_name]['layers'] = layers_mapping
            
        print(f"Processed network {network_name} with {len(yaml_files)} layers")
    
    # Save dictionary to a Python file, but manually write it with layers in correct order
    output_file = os.path.join(networks_dir, 'model_definitions.py')
    with open(output_file, 'w') as f:
        f.write("# Auto-generated network dictionary\n\n")
        f.write("MODEL_DEFINITIONS = {\n")

        # Write each network with manually sorted layers
        for network_idx, (network_name, network_data) in enumerate(network_dict.items()):
            f.write(f"    '{network_name}': {{\n")

            # Extract and sort layer indices
            layers = network_data.get('layers', {})
            sorted_layer_nums = sorted(layers.keys())

            # Write the 'layers' mapping
            f.write("        'layers': {\n")

            for idx, layer_num in enumerate(sorted_layer_nums):
                layer_dict = layers[layer_num]

                # Format the layer dictionary as string
                layer_str = pprint.pformat(layer_dict, indent=4, width=120)
                # Adjust indentation for nested level
                layer_str = '\n'.join(' ' * 12 + line for line in layer_str.split('\n'))

                f.write(f"            {layer_num}: {layer_str}")

                # Add comma if not the last layer
                if idx < len(sorted_layer_nums) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

            f.write("        }\n")

            # Close the network dictionary and add comma if not the last network
            if network_idx < len(network_dict) - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")

        # Close the main dictionary
        f.write("}\n")
    
    print(f"Converted {len(network_dict)} networks to {output_file}")
    
    # Print summary of layers for each network
    for network, layers in network_dict.items():
        print(f"Network: {network} - {len(layers)} layers")

if __name__ == "__main__":
    main()
