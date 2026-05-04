# Config: alexnet_hetero_center_imc
# Function: improved_mapper
# Iteration: 2
# Improvement - Total: +0.0%  Compute: +0.0%  Comm: +0.0%

def _improved_mapper(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
    """
    Improved mapper function that considers layer dependencies and load balancing.
    """
    mapping_failed = False
    failure_reason = None
    num_chiplets = len(system.chiplets)
    
    # Treat current_available_crossbars as capacity units (IMC: crossbars, CMOS: weights)
    remaining_capacity = current_available_crossbars.copy()

    layer_name = model_layer_info['name']
    crossbars_required = model_layer_info['crossbars_required']
    action = [0] * num_chiplets
    resources_remaining = crossbars_required
    percentage_remaining = 100

    # Only consider compute chiplets (exclude I/O chiplets) and honor allowed_chiplet_ids if provided
    chiplet_indices = [idx for idx in range(num_chiplets) 
                      if not system.is_io_chiplet(idx + 1)]
    allowed = None
    if isinstance(preference, dict):
        allowed = preference.get('allowed_chiplet_ids')
    if allowed:
        allowed_set = set(allowed)
        chiplet_indices = [idx for idx in chiplet_indices if (idx + 1) in allowed_set]

    # Calculate layer requirements for each chiplet
    layer_requirements = []
    for chiplet_idx in chiplet_indices:
        chiplet = system.chiplets[chiplet_idx]
        requirement = self._calculate_layer_requirements(model_layer_info, chiplet)
        layer_requirements.append((chiplet_idx, requirement))

    # Sort chiplets based on layer requirements and available capacity
    sorted_chiplets = sorted(layer_requirements, key=lambda x: (x[1], -remaining_capacity[x[0]]))

    for chiplet_idx, _ in sorted_chiplets:
        if resources_remaining <= 0:
            break
        
        available_units = remaining_capacity[chiplet_idx]
        if available_units <= 0:
            continue

        units_needed_full = self._calculate_layer_requirements(model_layer_info, system.chiplets[chiplet_idx])
        if units_needed_full <= 0: continue

        if crossbars_required > 0:
            units_needed_for_resources = math.ceil(resources_remaining * units_needed_full / crossbars_required)
        else:
            units_needed_for_resources = 0
        
        if available_units >= units_needed_for_resources:
            remaining_capacity[chiplet_idx] -= units_needed_for_resources
            alloc_percentage = percentage_remaining
            percentage_remaining = 0
            resources_remaining = 0
        else:
            alloc_percentage = (available_units * 100 / units_needed_full) if units_needed_full > 0 else 0
            remaining_capacity[chiplet_idx] = 0
            percentage_remaining -= alloc_percentage
            resources_remaining -= math.ceil(alloc_percentage * crossbars_required / 100)
        
        action[chiplet_idx] = alloc_percentage

    if (resources_remaining - 10) > 0:
        mapping_failed = True
        failure_reason = "INSUFFICIENT_MEMORY"

    sum_action = sum(action)
    if sum_action > 0 and not math.isclose(sum_action, 100, rel_tol=1):
         print(f"Warning: Layer '{layer_name}' action sum is {sum_action}, not 100. Action: {action}")

    return remaining_capacity, action, mapping_failed, failure_reason