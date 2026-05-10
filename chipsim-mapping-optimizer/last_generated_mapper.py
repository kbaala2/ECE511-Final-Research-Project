# Config: vgg16_hetero_imc_heavy
# Function: centroid_balanced_mapper
# Iteration: 3
# Improvement - Total: +0.7%  Compute: +1.0%  Comm: +0.7%

def _centroid_balanced_mapper(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
    """
    Activation‑centroid anchor, occupancy‑aware sorting, and largest‑remainder
    allocation to eliminate capacity waste and improve load balance.
    """
    chiplet_network = system.chiplet_network
    chiplets = system.chiplets
    mapping_failed = False
    failure_reason = None
    num_chiplets = len(chiplets)
    remaining_capacity = current_available_crossbars.copy()

    if shortest_paths is None:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(chiplet_network))

    layer_name = model_layer_info['name']
    crossbars_required = model_layer_info['crossbars_required']
    action = [0] * num_chiplets

    # occupancy and available memory in capacity units
    occupancy_percentages = []
    avail_memory_list = []
    for idx, chiplet in enumerate(chiplets):
        if self.system.is_io_chiplet(idx + 1):
            avail_memory_list.append(0)
            occupancy_percentages.append(100)
        else:
            unit_size = chiplet.get_capacity_unit_size()
            avail_memory = remaining_capacity[idx] * unit_size
            total_memory = chiplet.get_total_memory()
            occupancy = (total_memory - avail_memory) / total_memory * 100 if total_memory > 0 else 100
            avail_memory_list.append(avail_memory)
            occupancy_percentages.append(occupancy)

    # enforce allowed chiplet ids (if any)
    allowed = None
    if hasattr(self, 'preference') and isinstance(self.preference, dict):
        allowed = self.preference.get('allowed_chiplet_ids')
    if not allowed and isinstance(preference, dict):
        allowed = preference.get('allowed_chiplet_ids')

    # ---------- anchor selection ----------
    available_chiplet_ids = [idx + 1 for idx, cu in enumerate(remaining_capacity)
                             if cu > 0 and not self.system.is_io_chiplet(idx + 1)]
    if allowed:
        allowed_set = set(allowed)
        available_chiplet_ids = [cid for cid in available_chiplet_ids if cid in allowed_set]
    if not available_chiplet_ids:
        return remaining_capacity, None, True, "NO_AVAILABLE_CHIPLETS"

    if not layer_mappings:
        # first layer: chiplet with max available capacity (same as baseline)
        chiplet_info = []
        for chiplet_id in available_chiplet_ids:
            idx = chiplet_id - 1
            chiplet_info.append((chiplet_id, avail_memory_list[idx]))
        chiplet_info.sort(key=lambda x: -x[1])
        anchor = chiplet_info[0][0]
    else:
        # centroid weighted by activation size from previous layer
        prev_partitions = layer_mappings[-1][1]   # list of (chiplet_id, percentage)
        activation_size = model_layer_info.get('input_activation', 0)
        if activation_size is None or activation_size <= 0:
            activation_size = 1e-3   # fallback to unbiased average

        sum_x = sum_y = total_weight = 0
        for chiplet_id, pct in prev_partitions:
            weight = pct * activation_size
            total_weight += weight
            node = chiplet_network.nodes[chiplet_id]
            x = node.get('x', 0)
            y = node.get('y', 0)
            sum_x += x * weight
            sum_y += y * weight

        if total_weight > 0:
            cx = sum_x / total_weight
            cy = sum_y / total_weight
        else:
            # fallback: largest partition
            largest_id = max(prev_partitions, key=lambda x: x[1])[0]
            node = chiplet_network.nodes[largest_id]
            cx = node.get('x', 0)
            cy = node.get('y', 0)

        # anchor = available chiplet nearest to centroid
        best_dist = math.inf
        anchor = available_chiplet_ids[0]
        for cid in available_chiplet_ids:
            node = chiplet_network.nodes[cid]
            dx = node.get('x', 0) - cx
            dy = node.get('y', 0) - cy
            d = abs(dx) + abs(dy)          # Manhattan
            if d < best_dist:
                best_dist = d
                anchor = cid

    # ---------- candidate sorting ----------
    distances = shortest_paths.get(anchor, {})
    candidates = []
    for chiplet_id in available_chiplet_ids:
        dist = distances.get(chiplet_id, math.inf)
        occ = occupancy_percentages[chiplet_id - 1]
        candidates.append((dist, occ, chiplet_id))
    # sort by distance (primary), occupancy (ascending – less occupied first)
    candidates.sort(key=lambda x: (x[0], x[1]))

    # ---------- largest‑remainder allocation ----------
    percentage_remaining = 100
    # first pass: allocate floor percentages
    for dist, occ, chiplet_id in candidates:
        if percentage_remaining <= 0:
            break
        idx = chiplet_id - 1
        available_units = remaining_capacity[idx]
        if available_units <= 0:
            continue
        units_needed_full = self._calculate_layer_requirements(model_layer_info, chiplets[idx])
        if units_needed_full <= 0:
            continue

        # maximum integer percentage whose floor resource usage fits
        max_percent_float = (available_units * 100.0) / units_needed_full
        max_percent_int = int(max_percent_float)
        if max_percent_int <= 0:
            continue

        alloc = min(percentage_remaining, max_percent_int)
        if alloc > 0:
            used = math.floor(alloc * units_needed_full / 100)
            remaining_capacity[idx] -= used
            percentage_remaining -= alloc
            action[idx] += alloc

    # second pass: distribute remaining percentage unit by unit,
    # using ceil for resource consumption to fill gaps
    if percentage_remaining > 0:
        # sort again by occupancy (lightest first) to preserve load balance
        candidates_by_occ = sorted(candidates, key=lambda x: x[1])
        for dist, occ, chiplet_id in candidates_by_occ:
            if percentage_remaining <= 0:
                break
            idx = chiplet_id - 1
            available_units = remaining_capacity[idx]
            if available_units <= 0:
                continue
            units_needed_full = self._calculate_layer_requirements(model_layer_info, chiplets[idx])
            if units_needed_full <= 0:
                continue

            current_pct = action[idx]
            old_units_floor = math.floor(current_pct * units_needed_full / 100)
            # try adding 1%
            new_pct = current_pct + 1
            new_units_ceil = math.ceil(new_pct * units_needed_full / 100)
            extra_units = new_units_ceil - old_units_floor
            if extra_units <= remaining_capacity[idx]:
                remaining_capacity[idx] -= extra_units
                action[idx] += 1
                percentage_remaining -= 1

    if percentage_remaining > 0:
        # still some resources unallocated – mapping fails
        return remaining_capacity, None, True, "INSUFFICIENT_MEMORY"

    sum_action = sum(action)
    if not math.isclose(sum_action, 100.0, abs_tol=1e-6):
        # internal error protection
        pass

    return remaining_capacity, action, False, None