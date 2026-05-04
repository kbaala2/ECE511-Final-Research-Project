# Function name: improved_mapper
# Valid: True
# Validation retries: 0
# Runtime retries:    0
# Run succeeded:      True

def _improved_mapper(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
    """
    Improved mapper function that considers global optimization, data reuse, and dynamic mapping adjustment.
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

    # Calculate layer requirements for each chiplet
    layer_requirements = []
    for idx, chiplet in enumerate(system.chiplets):
        if self.system.is_io_chiplet(idx + 1):
            layer_requirements.append(0)
        else:
            layer_requirements.append(self._calculate_layer_requirements(model_layer_info, chiplet))

    # Sort chiplets based on their availability and layer requirements
    chiplet_info = []
    for idx, (chiplet_id, requirement) in enumerate(zip(range(1, num_chiplets + 1), layer_requirements)):
        if self.system.is_io_chiplet(chiplet_id):
            continue
        available_units = remaining_capacity[idx]
        if available_units <= 0:
            continue
        chiplet_info.append((chiplet_id, available_units, requirement))

    chiplet_info.sort(key=lambda x: (-x[1], x[2]))

    # Allocate resources to chiplets
    for chiplet_id, available_units, requirement in chiplet_info:
        if resources_remaining <= 0:
            break
        idx = chiplet_id - 1
        if available_units <= 0:
            continue

        if crossbars_required > 0:
            units_needed_for_resources = math.ceil(resources_remaining * requirement / crossbars_required)
        else:
            units_needed_for_resources = 0

        if available_units >= units_needed_for_resources:
            remaining_capacity[idx] -= units_needed_for_resources
            alloc_percentage = percentage_remaining
            percentage_remaining = 0
            resources_remaining = 0
        else:
            alloc_percentage = (available_units * 100 / requirement) if requirement > 0 else 0
            remaining_capacity[idx] = 0
            percentage_remaining -= alloc_percentage
            resources_remaining -= math.ceil(alloc_percentage * crossbars_required / 100)

        action[idx] += alloc_percentage

    if (resources_remaining - 10) > 0:
        print(f"⚠️ Layer '{layer_name}' allocation postponed: {resources_remaining} resources could not be allocated.")
        mapping_failed = True
        failure_reason = "INSUFFICIENT_MEMORY"
        return remaining_capacity, None, mapping_failed, failure_reason

    sum_action = sum(action)
    if sum_action > 0 and not math.isclose(sum_action, 100, rel_tol=1):
         print(f"Warning: Layer '{layer_name}' action sum is {sum_action}, not 100. Action: {action}")

    return remaining_capacity, action, mapping_failed, failure_reason