MAPPER_CONTRACT = """
STRICT REQUIREMENTS — your code MUST follow these rules exactly:

1. METHOD SIGNATURE: The method must be named based on the proposal you have chosen 
   and accept exactly these arguments:
   (self, model_layer_info, system, preference, 
    current_available_crossbars, layer_mappings=None, shortest_paths=None)

2. RETURN TYPE: Must return a tuple of exactly 4 values:
   (remaining_capacity, action, mapping_failed, failure_reason)
   - remaining_capacity: numpy array (copy of current_available_crossbars, modified)
   - action: list of floats, length = number of chiplets, percentages summing to ~100
             OR None if mapping failed
   - mapping_failed: bool
   - failure_reason: str or None

3. DO NOT MODIFY:
   - The argument list  
   - The return type structure
   - The _calculate_layer_requirements method (treat it as a black box)
   - Any System or Chiplet class interfaces

4. YOU MAY MODIFY:
   - How the starting chiplet is selected
   - How chiplets are sorted/prioritized for allocation
   - The greedy allocation loop logic
   - Adding look-ahead or global optimization within the method

5. AVAILABLE DATA you can use:
   - model_layer_info['name'], ['crossbars_required'], ['layer_type'], 
     ['filter_size'], ['out_channels'], ['in_features'], ['out_features']
   - system.chiplets (list of Chiplet objects)
   - system.chiplet_network (networkx graph)
   - system.is_io_chiplet(chiplet_id) -> bool
   - chiplet.get_capacity_unit_size(), chiplet.get_total_memory()
   - chiplet.type ('IMC' or 'CMOS'), chiplet.crossbar_rows, etc.
   - shortest_paths[src][dst] -> int (hop count)
   - layer_mappings: list of (layer_idx, [(chiplet_id, percentage)]) for prior layers

6. Output ONLY the complete method body. No explanation, no markdown fences,
   no imports (numpy and math are already imported).
"""