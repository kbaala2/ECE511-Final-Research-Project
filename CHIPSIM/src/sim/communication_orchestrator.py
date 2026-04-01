# communication_orchestrator.py

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import ast
from src.core.comm_types import ComputePhase


class CommunicationOrchestrator:
    """
    Orchestrates communication simulation for active models.
    
    Responsibilities:
    - Find active communication phases across all models
    - Create and populate traffic matrices
    - Track co-active traffic between models
    - Update phase latencies after simulation
    - Record DSENT statistics
    
    Args:
        comm_simulator: Communication simulator instance
        dsent_collector: Stats collector for DSENT results
        system: System instance for accessing chiplet information
    """
    
    def __init__(self, comm_simulator, dsent_collector, system):
        """
        Initialize the communication orchestrator.
        
        Args:
            comm_simulator: The communication simulator instance
            dsent_collector: StatsCollector for DSENT statistics
            system: System instance for chiplet topology information
        """
        self.comm_simulator = comm_simulator
        self.dsent_collector = dsent_collector
        self.system = system
        self.simulation_call_counter = 0
    
    def simulate_communication(
        self, 
        active_mapped_models: Dict,
        global_time_us: float
    ) -> bool:
        """
        Simulate communication for all active models during the current timestep.
        
        Finds all active communication phases across models, creates traffic matrices,
        tracks co-active traffic, runs the communication simulator, and updates results.
        
        Args:
            active_mapped_models: Dictionary of currently active mapped models
            global_time_us: Current simulation time in microseconds
            
        Returns:
            bool: True if simulation was successful, False otherwise
        """
        # Collect active communication phases and create traffic matrices
        active_phases, traffic_matrices_info = self._collect_active_phases_and_traffic(
            active_mapped_models, global_time_us
        )
        
        # If no active traffic, return early
        if not traffic_matrices_info:
            return False
        
        # Check if any phases need initial simulation
        if not self._needs_initial_simulation(active_phases, active_mapped_models):
            # Skip simulation if no new phases are starting since the external simulator
            # already tracks ongoing traffic internally
            return False
        
        # Prepare traffic matrices for simulator
        traffic_matrices_for_sim = self._prepare_traffic_matrices(traffic_matrices_info)
        
        # Track co-active traffic
        self.simulation_call_counter += 1
        co_active_phases = self._prepare_co_active_phases(
            traffic_matrices_info, active_mapped_models
        )
        self._record_co_active_traffic(
            traffic_matrices_info, active_mapped_models, 
            co_active_phases, global_time_us
        )
        
        # Run communication simulation
        print(f"\n🔹 Simulating communication for {len(active_phases)} active phases...")
        network_stats = self.comm_simulator.simulate_communication(
            traffic_matrices=traffic_matrices_for_sim,
            simulation_type="active"
        )
        
        # Update results and return status
        return self._update_simulation_results(
            network_stats, active_mapped_models, active_phases, global_time_us
        )
    
    def _collect_active_phases_and_traffic(
        self, 
        active_mapped_models: Dict,
        global_time_us: float
    ) -> Tuple[List, List]:
        """
        Find all active communication phases and create traffic matrices.
        
        Args:
            active_mapped_models: Dictionary of active mapped models
            global_time_us: Current simulation time
            
        Returns:
            Tuple of (active_phases_list, traffic_matrices_info_list)
        """
        active_network_input_phases = []
        traffic_matrices_info = []
        
        for model_idx, mapped_model in active_mapped_models.items():
            # Get active phases for this model
            active_phases_for_network = mapped_model.get_active_phases(global_time_us)
            
            # Filter out COMPUTE phases - we only simulate communication phases
            communication_phases = [
                phase_instance for phase_instance in active_phases_for_network
                if not isinstance(mapped_model.phases[phase_instance.phase_id], ComputePhase)
            ]
            
            # Process each active communication phase
            if communication_phases:
                for phase_instance in communication_phases:
                    phase_id = phase_instance.phase_id
                    input_idx = phase_instance.input_idx
                    phase = mapped_model.phases[phase_id]
                    layer_idx = getattr(phase, 'layer_idx', -1)
                    phase_type_name = phase.get_phase_type_name()
                    
                    # Use scaled traffic from the phase instance
                    layer_traffic = (phase_instance.scaled_traffic 
                                   if hasattr(phase_instance, 'scaled_traffic') 
                                   else phase.traffic)
                    
                    # Check if layer traffic is truly empty (no actual packets)
                    if not self._has_actual_traffic(layer_traffic):
                        print(f"⚠️ WARNING: No actual traffic (all zero or empty destinations) "
                              f"for model {model_idx}, input {input_idx}, phase {phase_id}, "
                              f"layer {layer_idx}, phase_type {phase_type_name}")
                        continue
                    
                    # Add to the list of active phases
                    active_network_input_phases.append(
                        (model_idx, input_idx, phase_id, layer_idx, phase_type_name)
                    )
                    
                    # Create traffic matrix for this phase
                    traffic_matrix = self._create_traffic_matrix(
                        layer_traffic, model_idx, input_idx, layer_idx
                    )
                    
                    # Store the info needed for simulation
                    traffic_matrices_info.append({
                        'matrix': traffic_matrix,
                        'model_idx': model_idx,
                        'input_idx': input_idx,
                        'phase_id': phase_id,
                        'layer_idx': layer_idx,
                        'phase_type': phase_type_name
                    })
        
        return active_network_input_phases, traffic_matrices_info
    
    def _has_actual_traffic(self, layer_traffic: Dict) -> bool:
        """
        Check if layer traffic contains any non-zero packets.
        
        Args:
            layer_traffic: Dictionary mapping source to destinations and amounts
            
        Returns:
            bool: True if there's actual traffic, False otherwise
        """
        if layer_traffic:
            for src, dests in layer_traffic.items():
                if dests and any(amount > 0 for amount in dests.values()):
                    return True
        return False
    
    def _create_traffic_matrix(
        self, 
        layer_traffic: Dict,
        model_idx: int,
        input_idx: int,
        layer_idx: int
    ) -> np.ndarray:
        """
        Create and populate a traffic matrix for a specific phase.
        
        Args:
            layer_traffic: Dictionary of traffic data {src: {dst: amount}}
            model_idx: Model index for error messages
            input_idx: Input index for error messages
            layer_idx: Layer index for error messages
            
        Returns:
            numpy array representing the traffic matrix
            
        Raises:
            ValueError: If invalid chiplet IDs are found
        """
        traffic_matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
        
        # Populate the traffic matrix
        for src_chiplet, destinations in layer_traffic.items():
            if 1 <= src_chiplet <= self.system.num_chiplets:
                for dst_chiplet, amount in destinations.items():
                    if 1 <= dst_chiplet <= self.system.num_chiplets:
                        # Ensure amount is non-negative after potential scaling
                        traffic_matrix[src_chiplet - 1, dst_chiplet - 1] = max(0, amount)
                    else:
                        raise ValueError(
                            f"Invalid destination chiplet ID {dst_chiplet} found in traffic data "
                            f"for model {model_idx}, input {input_idx}, layer {layer_idx}."
                        )
            else:
                raise ValueError(
                    f"Invalid source chiplet ID {src_chiplet} found in traffic data "
                    f"for model {model_idx}, input {input_idx}, layer {layer_idx}."
                )
        
        return traffic_matrix
    
    def _needs_initial_simulation(
        self, 
        active_phases: List,
        active_mapped_models: Dict
    ) -> bool:
        """
        Check if there's at least one phase that needs initial simulation.
        
        Args:
            active_phases: List of active phase tuples
            active_mapped_models: Dictionary of active mapped models
            
        Returns:
            bool: True if any phase needs initial simulation, False otherwise
        """
        for model_idx, input_idx, phase_id, layer_idx, phase_type_name in active_phases:
            network = active_mapped_models[model_idx]
            phase_instance = network.phase_instances[input_idx][phase_id]
            if phase_instance.latency_us == 0:
                return True
        return False
    
    def _prepare_traffic_matrices(self, traffic_matrices_info: List) -> List[Tuple]:
        """
        Prepare traffic matrices in the format expected by the simulator.
        
        Args:
            traffic_matrices_info: List of traffic matrix information dictionaries
            
        Returns:
            List of tuples in simulator format
        """
        traffic_matrices_for_sim = []
        for info in traffic_matrices_info:
            traffic_matrices_for_sim.append((
                info['matrix'],
                info['model_idx'],
                info['input_idx'],
                info['phase_id']  # Pass phase_id instead of layer_idx for uniqueness
            ))
        return traffic_matrices_for_sim
    
    def _prepare_co_active_phases(
        self, 
        traffic_matrices_info: List,
        active_mapped_models: Dict
    ) -> List[Dict]:
        """
        Prepare co-active phase information for tracking.
        
        Args:
            traffic_matrices_info: List of traffic matrix information dictionaries
            active_mapped_models: Dictionary of active mapped models
            
        Returns:
            List of dictionaries containing co-active phase information
        """
        co_active_phases = []
        
        for info in traffic_matrices_info:
            model_idx = info['model_idx']
            input_idx = info['input_idx']
            phase_id = info['phase_id']
            phase_type = info['phase_type']
            layer_idx = info['layer_idx']
            
            # Get the phase object to determine layer information
            mapped_model = active_mapped_models[model_idx]
            phase = mapped_model.phases[phase_id]
            
            # Determine layer information based on phase type
            if phase_type == 'ACTIVATION_COMM':
                layer_info = getattr(phase, 'layer_idx', -1)
                next_compute_layer = None
            elif phase_type == 'WEIGHT_LOADING_COMM':
                if hasattr(phase, 'layers_to_load') and phase.layers_to_load:
                    layer_info = phase.layers_to_load[0]
                    next_compute_layer = layer_info
                else:
                    layer_info = -1
                    next_compute_layer = -1
            else:
                layer_info = -1
                next_compute_layer = None
            
            # Get traffic contribution for this phase
            phase_instance = mapped_model.phase_instances[input_idx][phase_id]
            traffic_contribution = {}
            if hasattr(phase_instance, 'scaled_traffic') and phase_instance.scaled_traffic:
                traffic_contribution = phase_instance.scaled_traffic
            elif phase.traffic:
                traffic_contribution = phase.traffic
            
            co_active_phases.append({
                'model_idx': model_idx,
                'input_idx': input_idx,
                'phase_id': phase_id,
                'phase_type': phase_type,
                'layer_idx': layer_info,
                'next_compute_layer': next_compute_layer,
                'traffic_contribution': traffic_contribution
            })
        
        return co_active_phases
    
    def _record_co_active_traffic(
        self,
        traffic_matrices_info: List,
        active_mapped_models: Dict,
        co_active_phases: List[Dict],
        global_time_us: float
    ):
        """
        Record co-active traffic information in each model that has active phases.
        
        Args:
            traffic_matrices_info: List of traffic matrix information
            active_mapped_models: Dictionary of active mapped models
            co_active_phases: List of co-active phase information
            global_time_us: Current simulation time
        """
        models_with_active_phases = set(info['model_idx'] for info in traffic_matrices_info)
        for model_idx in models_with_active_phases:
            mapped_model = active_mapped_models[model_idx]
            mapped_model.record_co_active_traffic(
                simulation_call_id=self.simulation_call_counter,
                simulation_time_us=global_time_us,
                co_active_phases=co_active_phases
            )
    
    def _update_simulation_results(
        self,
        network_stats: Optional[Dict],
        active_mapped_models: Dict,
        active_phases: List,
        global_time_us: float
    ) -> bool:
        """
        Update phase latencies and DSENT statistics from simulation results.
        
        Args:
            network_stats: Results from communication simulator
            active_mapped_models: Dictionary of active mapped models
            active_phases: List of active phase tuples
            global_time_us: Current simulation time
            
        Returns:
            bool: True if update was successful
            
        Raises:
            RuntimeError: If communication simulation failed or returned invalid results
        """
        if not network_stats:
            raise RuntimeError(
                "❌ ERROR: Communication simulation returned no results.\n"
                "The network simulator (Garnet) failed to produce valid statistics."
            )
        
        if "latency" not in network_stats:
            raise RuntimeError(
                f"❌ ERROR: Communication simulation returned invalid results.\n"
                f"Missing 'latency' key in network statistics.\n"
                f"Stats received: {network_stats}"
            )
        
        # Process latencies for all phases
        if "all_phase_latencies_us" in network_stats["latency"]:
            self._update_phase_latencies(
                network_stats["latency"]["all_phase_latencies_us"],
                active_mapped_models,
                global_time_us
            )
        else:
            raise RuntimeError(
                f"❌ ERROR: Communication simulation returned zero latencies for all phases.\n"
                f"This indicates the network simulator failed to properly process traffic.\n"
                f"Stats received: {network_stats}"
            )
        
        # Store DSENT stats if available
        self._store_dsent_stats(network_stats, active_phases, global_time_us)
        
        return True
    
    def _update_phase_latencies(
        self,
        all_phase_latencies: Dict,
        active_mapped_models: Dict,
        global_time_us: float
    ):
        """
        Update latency metrics for all active phases.
        
        Args:
            all_phase_latencies: Dictionary of phase latencies from simulator
            active_mapped_models: Dictionary of active mapped models
            global_time_us: Current simulation time
        """
        for key, new_simulated_latency_us in all_phase_latencies.items():
            # Accept tuple or stringified tuple keys of the form (net_idx, input_idx, phase_id)
            if isinstance(key, tuple) and len(key) >= 3:
                net_idx, inp_idx, phase_id = key[0], key[1], key[2]
            elif isinstance(key, str):
                try:
                    parsed = ast.literal_eval(key)
                    if isinstance(parsed, tuple) and len(parsed) >= 3:
                        net_idx, inp_idx, phase_id = parsed[0], parsed[1], parsed[2]
                    else:
                        print(f"ERROR: Unexpected key format after parse: {key} -> {parsed}")
                        continue
                except Exception:
                    print(f"ERROR: Unexpected key format: {key} (type: {type(key)})")
                    continue
            else:
                print(f"ERROR: Unexpected key format: {key} (type: {type(key)})")
                continue
            
            if net_idx in active_mapped_models:
                network = active_mapped_models[net_idx]
                # Update phase latency using the phase-based method
                network.update_phase_latency(
                    inp_idx, phase_id, new_simulated_latency_us, global_time_us
                )
        
        # After updating all phase latencies, sync to layer-level metrics
        for network in active_mapped_models.values():
            network.sync_phase_metrics_to_layers()
    
    def _store_dsent_stats(
        self,
        network_stats: Dict,
        active_phases: List,
        global_time_us: float
    ):
        """
        Store DSENT power/energy/area statistics.
        
        Args:
            network_stats: Results from communication simulator
            active_phases: List of active phase tuples
            global_time_us: Current simulation time
        """
        # DEBUG DSENT: use dedicated logger in temp/logs (easy to remove later)
        logger = None
        try:
            logger = logging.getLogger("dsent_debug")
            if not logger.handlers:
                logs_dir = os.path.join(os.getcwd(), "temp", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(os.path.join(logs_dir, "dsent_debug.log"))
                fh.setLevel(logging.INFO)
                fmt = logging.Formatter("%(asctime)s | %(message)s")
                fh.setFormatter(fmt)
                logger.addHandler(fh)
        except Exception:
            logger = None

        # DEBUG DSENT: high-level visibility into what we got back from the simulator
        if logger:
            try:
                has_power = bool(network_stats.get('power'))
                has_area = bool(network_stats.get('area'))
                has_energy = bool(network_stats.get('energy'))
                logger.info(
                    f"_store_dsent_stats at {global_time_us} us: "
                    f"power={has_power}, area={has_area}, energy={has_energy}"
                )
            except Exception:
                pass

        if network_stats.get('power') or network_stats.get('area') or network_stats.get('energy'):
            dsent_results = {
                'power': network_stats.get('power', {}),
                'area': network_stats.get('area', {}),
                'energy': network_stats.get('energy', {})
            }
            
            # Store DSENT stats using StatsCollector
            if dsent_results['power'] or dsent_results['area'] or dsent_results['energy']:
                # Include latency block as well so downstream power/energy profiling
                # can use the actual communication runtime instead of inferring it
                # from successive global_time_us entries.
                latency_block = network_stats.get('latency', {})

                global_dsent_entry = {
                    'global_time_us': global_time_us,
                    'active_phases': [
                        (net_idx, inp_idx, phase_id, layer_idx, phase_type_name)
                        for net_idx, inp_idx, phase_id, layer_idx, phase_type_name in active_phases
                    ],
                    'latency': latency_block,
                    **dsent_results
                }

                # DEBUG DSENT: inspect entry size and target file
                if logger:
                    try:
                        stats_path = getattr(self.dsent_collector, 'stats_file_path', '<unknown>')
                        logger.info(
                            f"adding entry to collector (file={stats_path}, "
                            f"routers={len(dsent_results['power'].get('routers', []))}, "
                            f"link_power_keys={list(dsent_results['power'].get('links', {}).keys())})"
                        )
                    except Exception:
                        pass

                self.dsent_collector.add_stats(global_dsent_entry)

                # Force immediate flush so DSENT stats persist even if another part of the code
                # changes the collector's dump threshold (safe to remove once issue is resolved)
                try:
                    self.dsent_collector.dump_stats()
                except Exception:
                    if logger:
                        logger.warning("Failed to dump DSENT stats immediately after add_stats()", exc_info=True)
        else:
            # Make this a warning
            if logger:
                try:
                    logger.warning("No DSENT stats found after simulation.")
                except Exception:
                    pass

