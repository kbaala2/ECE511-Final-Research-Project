# Function name: improved_mapper
# Valid: True
# Validation retries: 0
# Runtime retries:    0
# Run succeeded:      True

def _improved_mapper(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
    """
    Improved mapper that considers layer dependencies, chiplet capacity, and performance/energy efficiency metrics.
    """
    chiplet_network = system.chiplet_network
    chiplets = system.chiplets
    
    mapping_failed = False
    failure_reason = None
    num_chiplets = len(chiplets)
    # Treat current_available_crossbars as capacity units (IMC: crossbars, CMOS: weights)
    remaining_capacity = current_available_crossbars.copy()

    if shortest_paths is None:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(chiplet_network))
    
    active_metric = next((metric for metric, active in preference.items() if active), "performance")
            
    layer_name = model_layer_info['name']
    crossbars_required = model_layer_info['crossbars_required']
    action = [0] * num_chiplets
    resources_remaining = crossbars_required
    percentage_remaining = 100

    # Consider layer dependencies and map closely connected layers to nearby chiplets
    if layer_mappings:
        last_layer_mapping = layer_mappings[-1][1]
        last_chiplet = max(last_layer_mapping, key=lambda x: x[1])[0]
        starting_chiplet = last_chiplet
    else:
        # Start with the chiplet that has the most available capacity
        available_chiplet_ids = [idx + 1 for idx, cu in enumerate(remaining_capacity) 
                                if cu > 0 and not self.system.is_io_chiplet(idx + 1)]
        if not available_chiplet_ids:
            return remaining_capacity, None, True, "NO_AVAILABLE_CHIPLETS"

        chiplet_info = []
        for chiplet_id in available_chiplet_ids:
            idx = chiplet_id - 1
            metric_value = chiplet_network.nodes[chiplet_id].get(active_metric, 0)
            chiplet_info.append((chiplet_id, remaining_capacity[idx], metric_value))

        chiplet_info.sort(key=lambda x: (-x[1], -x[2]))
        
        sorted_chiplets_ids = [chiplet_id for chiplet_id, _, _ in chiplet_info]
        starting_chiplet = sorted_chiplets_ids[0]

    distances = shortest_paths.get(starting_chiplet, {})
    if not distances and num_chiplets > 1:
        return remaining_capacity, None, True, "NO_CHIPLET_CONNECTIONS"

    available_chiplet_ids = [idx + 1 for idx, cu in enumerate(remaining_capacity) 
                            if cu > 0 and not self.system.is_io_chiplet(idx + 1)]
    # Apply allowed filter consistently
    allowed = None
    if hasattr(self, 'preference') and isinstance(self.preference, dict):
        allowed = self.preference.get('allowed_chiplet_ids')
    if allowed:
        allowed_set = set(allowed)
        available_chiplet_ids = [cid for cid in available_chiplet_ids if cid in allowed_set]
        
    chiplet_info = []
    for chiplet_id in available_chiplet_ids:
        idx = chiplet_id - 1
        distance = distances.get(chiplet_id, math.inf)
        metric_value = chiplet_network.nodes[chiplet_id].get(active_metric, 0)
        chiplet_info.append((chiplet_id, distance, remaining_capacity[idx], metric_value))

    chiplet_info.sort(key=lambda x: (x[1], -x[2], -x[3]))
    
    sorted_chiplets_ids = [chiplet_id for chiplet_id, _, _, _ in chiplet_info]
    
    for chiplet_id in sorted_chiplets_ids:
        if resources_remaining <= 0:
            break
        chiplet_idx = chiplet_id - 1
        available_units = remaining_capacity[chiplet_idx]
        if available_units <= 0:
            continue

        units_needed_full = self._calculate_layer_requirements(model_layer_info, chiplets[chiplet_idx])
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
        
        action[chiplet_idx] += alloc_percentage

    if (resources_remaining - 10) > 0:
        return remaining_capacity, None, True, "INSUFFICIENT_MEMORY"

    sum_action = sum(action)
    if sum_action > 0 and not math.isclose(sum_action, 100, rel_tol=1):
         print(f"Warning: Layer '{layer_name}' action sum is {sum_action}, not 100. Action: {action}")
    
    return remaining_capacity, action, False, None