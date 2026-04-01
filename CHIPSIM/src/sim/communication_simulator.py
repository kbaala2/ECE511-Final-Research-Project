import os
import sys
import subprocess
import gzip
import re
import json
import tempfile
import hashlib

import numpy as np
import logging

from src.managers.cache_manager import CacheManager

class CommunicationSimulator:
    """
    Handles communication simulation using Garnet or Booksim with caching.
    """

    # ====================================================
    # Initialization
    # ====================================================
    def __init__(self, 
                 system, 
                 simulator_type="Garnet", 
                 communication_method="non-pipelined", 
                 cache_file="communication_cache.pkl", 
                 clear_cache=False, 
                 network_operation_frequency=1000000000,
                 enable_dsent=True,
                 enable_cache=False,
                 dsent_tech_node="32",
                 gem5_sim_cycles=500000000,
                 gem5_injection_rate=0.0,
                 gem5_ticks_per_cycle=1000,
                 gem5_deadlock_threshold=None,
                 gem5_topology_config_relpath="configs/topologies/myTopology.yaml"):
        """
        Initialize the communication simulator.
        
        Args:
            system: System object containing topology information
            simulator_type (str): Type of simulator to use ("Garnet" or "Booksim")
            communication_method (str): Communication method ("pipelined" or "non-pipelined")
            cache_file (str): Path to the cache file
            clear_cache (bool): Whether to clear the existing cache
            network_operation_frequency (int): Network operation frequency in Hz
            enable_dsent (bool): Whether to run DSENT simulation for power/area
        """
        self.system = system
        self.simulator_type = simulator_type
        self.communication_method = communication_method
        self.network_operation_frequency = network_operation_frequency
        self.dsent_tech_node = dsent_tech_node
        self.enable_dsent = enable_dsent
        self.enable_cache = enable_cache

        # gem5 runtime configuration
        self.gem5_sim_cycles = gem5_sim_cycles
        self.gem5_injection_rate = gem5_injection_rate
        self.gem5_ticks_per_cycle = gem5_ticks_per_cycle
        self.gem5_deadlock_threshold = gem5_deadlock_threshold
        self.gem5_topology_config_relpath = gem5_topology_config_relpath
        
        # Set the gem5 path as a self attribute (absolute path)
        # Get the directory where this file is located, then navigate to integrations/gem5
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        chiplet_simulator_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
        self.gem5_path = os.path.join(chiplet_simulator_root, "integrations", "gem5")
        
        # Verify Garnet standalone is built if using Garnet simulator
        if self.simulator_type == "Garnet":
            self._verify_garnet_availability()
        
        # Initialize the cache manager
        self.cache_manager = CacheManager(cache_file, clear_cache)

        # Setup debug logger (once)
        self._comm_logger = logging.getLogger('comm_debug')
        if not self._comm_logger.handlers:
            self._comm_logger.setLevel(logging.INFO)
            logs_dir = os.path.join(os.getcwd(), 'temp', 'logs')
            try:
                os.makedirs(logs_dir, exist_ok=True)
            except Exception:
                pass
            fh = logging.FileHandler(os.path.join(logs_dir, 'comm_debug.log'))
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s | %(message)s')
            fh.setFormatter(fmt)
            self._comm_logger.addHandler(fh)

        # Convert adjacency matrix to Garnet topology file for AnyNet
        if not hasattr(self.system, 'adj_matrix') or self.system.adj_matrix is None:
            raise ValueError("system.adj_matrix not found or is None. An adjacency matrix is required for AnyNet topology.")
        
        # Host-side path for writing the topology YAML used by gem5
        topology_yaml_path_host = os.path.join(self.gem5_path, "configs/topologies/myTopology.yaml")
        self._convert_adj_matrix_to_yaml_topology(self.system.adj_matrix, topology_yaml_path_host)

    # ====================================================
    # Garnet Availability Check
    # ====================================================
    def _verify_garnet_availability(self):
        """
        Verify that Garnet standalone is built and available.
        
        Raises:
            FileNotFoundError: If gem5 directory or Garnet binary is not found
            RuntimeError: If Garnet standalone is not built
        """
        # Check if gem5 directory exists
        if not os.path.isdir(self.gem5_path):
            raise FileNotFoundError(
                f"❌ gem5 directory not found at: {self.gem5_path}\n"
                f"Please see docs/gem5-integration.md for installation instructions."
            )
        
        # Check if Garnet standalone binary exists
        garnet_binary = os.path.join(self.gem5_path, "build", "Garnet_standalone", "gem5.opt")
        
        if not os.path.isfile(garnet_binary):
            raise RuntimeError(
                f"❌ Garnet standalone is not installed.\n"
                f"Please see docs/gem5-integration.md for build instructions."
            )
        
        print(f"✅ Garnet standalone found at: {garnet_binary}")

    # ====================================================
    # Main Communication Simulation
    # ====================================================
    def simulate_communication(self, traffic_matrices, simulation_type="active", network_name=None):
        """
        Simulate communication using multiple traffic matrices.
        
        Args:
            traffic_matrices: List of tuples (traffic_matrix, network_idx, input_idx, phase_id)
            simulation_type (str): Type of simulation ("active", "aggregate", etc.)
            network_name (str): Optional network name for logging
            
        Returns:
            dict: Network simulation statistics with the following structure:
                {
                    'latency': {
                        'simTicks': float,
                        'all_phase_latencies': {  # In ticks
                            (network_idx, input_idx, phase_id): latency_ticks,
                            ...
                        },
                        'all_phase_latencies_us': {  # In microseconds
                            (network_idx, input_idx, phase_id): latency_us,
                            ...
                        },
                        'average_packet_latency': float,
                        'total_runtime_us': float,
                        ...
                    },
                    'power': {...},  # DSENT power statistics
                    'area': {...},   # DSENT area statistics
                    'energy': {...}  # DSENT energy statistics
                }
        """
        # ----------------------------------------------------
        # Check for empty traffic
        # ----------------------------------------------------
        if not traffic_matrices:
            print(f"No traffic found for {simulation_type} simulation.")
            return {}
        
        # ----------------------------------------------------
        # Generate cache key
        # ----------------------------------------------------
        # Build robust system params for cache key
        # Include topology hash and DSENT enable flag to avoid cross-run collisions
        topo_hash = hashlib.sha256(self.system.adj_matrix.astype(int).tobytes()).hexdigest() if hasattr(self.system, 'adj_matrix') else "no_topo"
        system_params = {
            "num_chiplets": self.system.num_chiplets,
            "simulator_type": self.simulator_type,
            "communication_method": self.communication_method,
            "network_operation_frequency": self.network_operation_frequency,
            "gem5_sim_cycles": self.gem5_sim_cycles,
            "gem5_injection_rate": self.gem5_injection_rate,
            "gem5_ticks_per_cycle": self.gem5_ticks_per_cycle,
            "gem5_deadlock_threshold": self.gem5_deadlock_threshold,
            "dsent_tech_node": self.dsent_tech_node,
            "enable_dsent": bool(self.enable_dsent),
            "topology_hash": topo_hash,
            "topology_rows": 10,  # current Mesh_XY rows setting
        }
        
        # Combine all matrices for cache key calculation
        combined_traffic = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
        active_phases_info = []
        for traffic_matrix, network_idx, input_idx, phase_id in traffic_matrices:
            combined_traffic += traffic_matrix
            active_phases_info.append((network_idx, input_idx, phase_id))
        
        # Include active phases in cache key for uniqueness
        if active_phases_info:
            system_params["active_phases"] = sorted(active_phases_info)
        
        cache_key = self.cache_manager.compute_cache_key(
            traffic_matrix=combined_traffic,
            simulation_type=simulation_type,
            system_params=system_params
        )
        
        # ----------------------------------------------------
        # Check cache
        # ----------------------------------------------------
        cached_result = None
        if (self.enable_cache) and self.cache_manager.has_result(cache_key):
            cached_result = self.cache_manager.get_result(cache_key)

        if cached_result:
            # Validate that the cached result contains latencies for all active phases
            valid_cache = True
            if 'latency' in cached_result and 'all_phase_latencies_us' in cached_result['latency']:
                cached_keys = set(cached_result['latency']['all_phase_latencies_us'].keys())
                current_keys = set(active_phases_info)
                if not current_keys.issubset(cached_keys):
                    valid_cache = False
                    print(f"⚠️  Cached result is missing {len(current_keys - cached_keys)} phase(s). Rerunning simulation.")
            else:
                valid_cache = False
                print("⚠️  Cached result is incomplete. Rerunning simulation.")

            if valid_cache:
                network_info = f" (Network: {network_name})" if network_name else ""
                phases_info = f" ({len(active_phases_info)} active phases)" if active_phases_info else ""
                print(f"🔄 Using cached {simulation_type} communication simulation result{network_info}{phases_info}")
                return cached_result
        
        # ----------------------------------------------------
        # Prepare for simulation
        # ----------------------------------------------------
        total_packets = 0
        for traffic_matrix, _, _, _ in traffic_matrices:
            # Use total flits (sum of matrix values) to avoid truncating traces in gem5
            total_packets += int(np.sum(traffic_matrix))
            
        if self.simulator_type == "Garnet":
            # Prepare traffic files with multiple matrices and phase info
            traffic_file_txt, traffic_file_gz = self._prepare_garnet_traffic_files(traffic_matrices, simulation_type)
            
            # Check if there's actually traffic to simulate
            if traffic_file_txt is None or traffic_file_gz is None:
                # Return zero latency for all phases when there's no traffic
                network_stats = {
                    'latency': {
                        'all_phase_latencies_us': {
                            (network_idx, input_idx, phase_id): 0.0
                            for _, network_idx, input_idx, phase_id in traffic_matrices
                        }
                    }
                }
                return network_stats
            
            # Build simulation command
            simulation_command = self._build_garnet_simulation_command(
                traffic_file_gz, 
                total_packets
            )
            # Run the simulation
            network_stats = self._run_garnet_simulation(simulation_command)
            if network_stats:
                garnet_stats = network_stats
                network_stats = {'latency': garnet_stats}
                
                if self.enable_dsent:
                    # Run DSENT for power/area estimation
                    dsent_stats = self._run_dsent_simulation()
                    if dsent_stats:
                        # Power, Area, Energy from DSENT - with safe key access
                        power_stats = {'routers': [], 'links': {}, 'totals': {}}
                        area_stats = {'routers': [], 'links': {}, 'totals': {}}
                        energy_stats = {'routers': [], 'links': {}}

                        # Process totals if available
                        if 'totals' in dsent_stats:
                            power_stats['totals'] = {k:v for k,v in dsent_stats['totals'].items() if 'power' in k}
                            area_stats['totals'] = {k:v for k,v in dsent_stats['totals'].items() if 'area' in k}

                        # Process router results if available
                        if 'routers' in dsent_stats:
                            for r_res in dsent_stats['routers']:
                                router_id = r_res.get('router_id', 'unknown')
                                p, a, e = {}, {}, {}
                                
                                if 'results' in r_res:
                                    for section, values in r_res['results'].items():
                                        a_section = {}
                                        for k, v in values.items():
                                            if 'power' in k.lower():
                                                if section not in p: p[section] = {}
                                                p[section][k] = v
                                            if section.lower() == 'area':
                                                a_section[k] = v
                                            if 'energy' in k.lower():
                                                if section not in e: e[section] = {}
                                                e[section][k] = v
                                        if a_section:
                                            a[section] = a_section
                                
                                power_stats['routers'].append({'router_id': router_id, **p})
                                area_stats['routers'].append({'router_id': router_id, **a})
                                energy_stats['routers'].append({'router_id': router_id, **e})
                        
                        # Process link results if available
                        link_p, link_e = {}, {}
                        if 'links' in dsent_stats:
                            for section, values in dsent_stats['links'].items():
                                for k, v in values.items():
                                    if 'power' in k.lower():
                                        if section not in link_p: link_p[section] = {}
                                        link_p[section][k] = v
                                    if 'energy' in k.lower():
                                        if section not in link_e: link_e[section] = {}
                                        link_e[section][k] = v
                        power_stats['links'] = link_p
                        energy_stats['links'] = link_e
                        
                        network_stats['power'] = power_stats
                        network_stats['area'] = area_stats
                        network_stats['energy'] = energy_stats
                    else:
                        raise RuntimeError("❌ DSENT simulation failed. Exiting simulation.")
                else:
                    # DSENT disabled, populate with empty dicts
                    network_stats['power'] = {}
                    network_stats['area'] = {}
                    network_stats['energy'] = {}
        elif self.simulator_type == "Booksim":
            # TODO: Update Booksim to handle phase-based traffic
            # For Booksim we would need to implement support for multiple matrices
            # For now, let's combine them
            combined_matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
            for traffic_matrix, _, _, _ in traffic_matrices:
                combined_matrix += traffic_matrix
            network_stats = self._run_booksim_simulation(combined_matrix)
        else:
            raise ValueError(f"Unknown simulator type: {self.simulator_type}")
        
        # ----------------------------------------------------
        # Cache results
        # ----------------------------------------------------
        if network_stats:
            # Add total runtime stats
            if 'latency' in network_stats and 'simTicks' in network_stats['latency']:
                sim_ticks = network_stats['latency']['simTicks']
                
                network_stats["latency"]["total_runtime_s"] = sim_ticks / (self.network_operation_frequency * self.gem5_ticks_per_cycle)
                network_stats["latency"]["total_runtime_us"] = network_stats["latency"]["total_runtime_s"] * 1e6
                
                # Process all phase latencies if available
                if 'all_phase_latencies' in network_stats['latency']:
                    all_phase_latencies_us = {}
                    for (net_idx, inp_idx, phase_id), latency_ticks in network_stats['latency']['all_phase_latencies'].items():
                        # Convert ticks to microseconds
                        latency_us = latency_ticks / (self.network_operation_frequency * self.gem5_ticks_per_cycle) * 1e6
                        all_phase_latencies_us[(net_idx, inp_idx, phase_id)] = latency_us
                        msg2 = (f"Network {net_idx}, Input {inp_idx}, Phase {phase_id}: "
                                f"{latency_ticks:.0f} ticks -> {latency_us:.2f} μs")
                        print(f"  {msg2}")
                        try:
                            self._comm_logger.info(msg2)
                        except Exception:
                            pass
                    
                    # Store the converted latencies
                    network_stats["latency"]["all_phase_latencies_us"] = all_phase_latencies_us
                    
            # Save to cache when caching is enabled
            if (self.enable_cache):
                self.cache_manager.store_result(cache_key, network_stats)
        
        return network_stats

    def simulate_model_phase_communication(self, model):
        """
        Build traffic matrices for each communication phase of a single model and
        run simulate_communication to get per-phase latencies in microseconds.

        Args:
            model: MappedModel instance

        Returns:
            Dict[(net_idx, input_idx, phase_id), latency_us]
        """
        traffic_matrices = []
        net_idx = getattr(model, 'model_idx', 0)
        # Build per-phase traffic from model phases; use stored phase.traffic
        for phase_id, phase in model.phases.items():
            if hasattr(phase, 'traffic') and phase.traffic:
                # Convert dict traffic to dense matrix
                matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
                for src_chiplet_id, dests in phase.traffic.items():
                    for dst_chiplet_id, amount in dests.items():
                        if amount <= 0:
                            continue
                        matrix[src_chiplet_id - 1, dst_chiplet_id - 1] += amount

                # Skip phases with no actual traffic to avoid cache expecting missing latencies
                if np.count_nonzero(matrix) == 0:
                    continue

                # Push one tuple per input index for this phase (traffic identical across inputs)
                for input_idx in range(model.num_inputs):
                    traffic_matrices.append((matrix, net_idx, input_idx, phase_id))

        if not traffic_matrices:
            return {}

        sim_results = self.simulate_communication(traffic_matrices, simulation_type="individual_model", network_name=getattr(model, 'model_name', None))
        # Normalize return: extract all_phase_latencies_us if present
        lat_us = {}
        if sim_results and 'latency' in sim_results and 'all_phase_latencies_us' in sim_results['latency']:
            lat_us = sim_results['latency']['all_phase_latencies_us']
        return lat_us

    # ====================================================
    # Garnet Traffic File Preparation
    # ====================================================
    def _prepare_garnet_traffic_files(self, traffic_matrices, simulation_type):
        """
        Prepare traffic files for Garnet simulation with phase information.
        
        Args:
            traffic_matrices: List of tuples (traffic_matrix, network_idx, input_idx, phase_id)
            simulation_type (str): Type of simulation
            
        Returns:
            tuple: Paths to the text and gzipped traffic files
        """
        # ----------------------------------------------------
        # File setup
        # ----------------------------------------------------
        traffic_file_name = f"garnet_{simulation_type}_traffic"
        temp_dir = "temp"
        garnet_traffic_dir = os.path.join(temp_dir, "garnet_traffic")
        traffic_file_txt = os.path.join(garnet_traffic_dir, f"{traffic_file_name}.txt")
        traffic_file_gz = os.path.join(garnet_traffic_dir, f"{traffic_file_name}.gz")
        
        # Create directories if needed
        os.makedirs(garnet_traffic_dir, exist_ok=True)
        
        # Delete any existing files in the garnet_traffic directory before creating new ones
        if os.path.exists(garnet_traffic_dir):
            for filename in os.listdir(garnet_traffic_dir):
                file_path = os.path.join(garnet_traffic_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # ----------------------------------------------------
        # Convert traffic matrices to Garnet format
        # ----------------------------------------------------
        garnet_traffic_data = []
        garnet_traffic_data.append("# cycle i_ni i_router o_ni o_router vnet flits network_idx input_idx phase_id")
        
        # Process all traffic matrices
        traffic_data = []
        for traffic_matrix, network_idx, input_idx, phase_id in traffic_matrices:
            non_zero_count = np.count_nonzero(traffic_matrix)
            total_packets = np.sum(traffic_matrix)
            msg = (f"Matrix for network={network_idx}, input={input_idx}, phase={phase_id}: "
                   f"shape={traffic_matrix.shape}, non-zero entries={non_zero_count}, total packets={total_packets:.0f}")
            print(f"  {msg}")
            try:
                self._comm_logger.info(msg)
            except Exception:
                pass
            for i in range(traffic_matrix.shape[0]):
                for j in range(traffic_matrix.shape[1]):
                    if traffic_matrix[i, j] > 0:
                        src_chiplet_id = i
                        dst_chiplet_id = j
                        traffic_data.append((src_chiplet_id, dst_chiplet_id, traffic_matrix[i, j], network_idx, input_idx, phase_id))
        
        # One line per source-destination pair
        
        # Check if there's actually any traffic to simulate
        if len(traffic_data) == 0:
            # Return empty paths to indicate no simulation needed
            return None, None
        
        for line in traffic_data:
            source_pe = line[0]
            dest_pe = line[1]
            num_packets = int(line[2])
            network_idx = line[3]
            input_idx = line[4]
            phase_id = line[5]
            
            # Generate a single row with flits = num_packets and source info
            garnet_row = f"1 {source_pe} {source_pe} {dest_pe+100} {dest_pe} 0 {num_packets} {network_idx} {input_idx} {phase_id}"
            garnet_traffic_data.append(garnet_row)
        
        
        # Sort the garnet traffic data by clock cycle
        # Skip the first row (header) when sorting
        header = garnet_traffic_data[0]
        data_rows = garnet_traffic_data[1:]
        data_rows.sort(key=lambda x: int(x.split()[0]))
        garnet_traffic_data = [header] + data_rows
        
        # ----------------------------------------------------
        # Write and compress
        # ----------------------------------------------------
        with open(traffic_file_txt, 'w') as f:
            for line in garnet_traffic_data:
                f.write(f"{line}\n")

        # Compress the file
        with open(traffic_file_txt, 'rb') as f_in:
            with gzip.open(traffic_file_gz, 'wb') as f_out:
                f_out.write(f_in.read())
        
        return traffic_file_txt, traffic_file_gz

    # ====================================================
    # Garnet Simulation Command Builder
    # ====================================================
    def _build_garnet_simulation_command(self, traffic_file, max_packets):
        """
        Build the Garnet simulation command.
        
        Args:
            traffic_file (str): Path to the traffic file
            max_packets (int): Maximum number of packets
            
        Returns:
            str: Simulation command
        """
        # cmd = (
        #     f"{self.gem5_path}/build/Garnet_standalone/gem5.opt {self.gem5_path}/configs/example/garnet_synth_traffic.py "
        #     f"--num-cpus={self.system.num_chiplets} "
        #     f"--num-dirs={self.system.num_chiplets} "
        #     f"--topology=AnyNET_XY "
        #     f"--config-file={self.gem5_topology_config_relpath} "
        #     f"--sim-cycles={self.gem5_sim_cycles} "
        #     f"--injectionrate={self.gem5_injection_rate} "
        #     f"--network-trace-enable "
        #     f"--network-trace-file='../../{traffic_file}' "
        #     f"--network-trace-max-packets={max_packets-1} "
        # )
        cmd = (
            f"build/Garnet_standalone/gem5.opt configs/example/garnet_synth_traffic.py "
            # f"--num-cpus=100 "
            # f"--num-dirs=100 "
            # f"--topology=Mesh_XY "
            # f"--mesh-rows=10 "
            f"--num-cpus={self.system.num_chiplets} "
            f"--num-dirs={self.system.num_chiplets} "
            f"--topology=AnyNET_XY "
            f"--config-file=configs/topologies/myTopology.yaml "
            f"--sim-cycles={self.gem5_sim_cycles} "
            f"--injectionrate={self.gem5_injection_rate} "
            f"--network-trace-enable "
            f"--network-trace-file='../../{traffic_file}' "
            f"--network-trace-max-packets={max_packets-1} "
        )
        if self.gem5_deadlock_threshold is not None:
            cmd += f"--garnet-deadlock-threshold={self.gem5_deadlock_threshold}"
        return cmd

    # ====================================================
    # Garnet Simulation Runner
    # ====================================================
    def _run_garnet_simulation(self, command):
        """
        Run a Garnet simulation using the specified command.
        
        Args:
            command (str): Simulation command
            
        Returns:
            dict: Simulation results
        """
        # Save current directory
        current_dir = os.getcwd()
        
        print(f"Switching to directory: {self.gem5_path}")
        print("🔄 Running Garnet simulation...")
        
        try:
            # Change to gem5 directory and run command
            os.chdir(self.gem5_path)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            # Check for errors
            if process.returncode != 0:
                error_message = stderr.decode('utf-8')
                print(f"Error running Garnet simulation: {error_message}")
                return {}
            
            # Debug: Print stderr even if successful to see any warnings
            if stderr:
                pass # No output for now
                #print(f"Garnet stderr output: {stderr.decode('utf-8')}")
            
            # Read the simulation stats
            stats_file_path = "./m5out/stats.txt"
            print("📊 Extracting simulation results...")
            return self._extract_garnet_stats(stats_file_path)
            
        finally:
            # Always return to original directory
            os.chdir(current_dir)

    # ====================================================
    # Garnet Stats Extraction
    # ====================================================
    def _extract_garnet_stats(self, stats_file_path):
        """
        Extract statistics from a Garnet stats file.
        
        Args:
            stats_file_path (str): Path to the stats file
            
        Returns:
            dict: Extracted statistics
        """
        network_stats = {}
        
        # Extract latencies for all traffic sets
        all_phase_latencies = {}
        
        try:
            with open(stats_file_path, 'r') as stats_file:
                for line in stats_file:
                    line = line.strip()
                    
                    # Extract key statistics
                    if line.startswith("simTicks"):
                        parts = line.split()
                        network_stats["simTicks"] = float(parts[1])
                        
                    elif "system.ruby.network.average_packet_latency" in line:
                        parts = line.split()
                        network_stats["average_packet_latency"] = float(parts[1])
                    
                    elif "system.ruby.network.average_packet_queueing_latency" in line:
                        parts = line.split()
                        network_stats["average_packet_queueing_latency"] = float(parts[1])
                    
                    elif "system.ruby.network.average_packet_network_latency" in line:
                        parts = line.split()
                        network_stats["average_packet_network_latency"] = float(parts[1])
                    
                    elif "system.ruby.network.packets_injected::total" in line:
                        parts = line.split()
                        network_stats["packets_injected"] = float(parts[1])
                    
                    elif "system.ruby.network.packets_received::total" in line:
                        parts = line.split()
                        network_stats["packets_received"] = float(parts[1])
                    
                    elif "system.ruby.network.average_hops" in line:
                        parts = line.split()
                        network_stats["average_hops"] = float(parts[1])
                    
                    # Parse individual phase latencies
                    elif "system.ruby.network.phase_latency_" in line:
                        # Extract network_idx, input_idx, phase_id from the stat name
                        # Format: system.ruby.network.phase_latency_<net>_<inp>_<phase>
                        match = re.search(r'phase_latency_(\d+)_(\d+)_(\d+)', line)
                        if match:
                            net_idx = int(match.group(1))
                            inp_idx = int(match.group(2))
                            phase_id = int(match.group(3))
                            parts = line.split()
                            latency = float(parts[1])
                            all_phase_latencies[(net_idx, inp_idx, phase_id)] = latency
            
            # Add all_phase_latencies to network_stats
            if all_phase_latencies:
                network_stats["all_phase_latencies"] = all_phase_latencies
            
            return network_stats
        
        except Exception as e:
            print(f"Error reading stats file: {e}")
            return {}
        
    # ====================================================
    # DSENT Simulation
    # ====================================================
    def _run_dsent_simulation(self):
        """
        Run a DSENT simulation for power and area analysis by spawning a separate process.
        This isolates DSENT's memory usage and prevents memory leaks from affecting
        the main simulator process.
        """
        print("⚡️ Running DSENT for power and area analysis (in separate process)...")
        current_dir = os.getcwd()

        # The runner script is in the integrations directory
        runner_script_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "integrations", "dsent_runner.py"
        )
        if not os.path.isfile(runner_script_path):
            print(f"❌ CRITICAL ERROR: DSENT runner script not found at {runner_script_path}")
            sys.exit(1)

        # Create a temporary file for the results. It will be deleted automatically.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", dir=current_dir) as tmp_output_file:
            output_file_path = tmp_output_file.name

        try:
            # Command to execute the runner script
            command = [
                sys.executable,
                runner_script_path,
                "--gem5-dir", self.gem5_path,
                "--tech-node", self.dsent_tech_node,
                "--output-file", output_file_path
            ]
            
            # Execute the script. It will handle changing directories itself.
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True  # Raises CalledProcessError on non-zero exit codes
            )
            
            # For debugging, print stdout/stderr from the runner
            # print(process.stdout)
            # if process.stderr:
            #     print(process.stderr, file=sys.stderr)
            
            # Read results from the temporary file
            with open(output_file_path, 'r') as f:
                results = json.load(f)
            
            # If results are empty, it means the runner failed.
            if not results:
                print("❌ DSENT runner script finished with errors but did not raise an exception.")
                return {}

            return results

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running DSENT runner script.")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            return {}
        except FileNotFoundError:
            print(f"❌ Error: Could not find the DSENT results file at {output_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Error: Could not decode JSON from DSENT results file at {output_file_path}")
            return {}
        finally:
            # Clean up the temporary file
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    # ====================================================
    # Booksim Simulation Runner
    # ====================================================
    def _run_booksim_simulation(self, traffic_matrix):
        """
        Run a Booksim simulation.

        Args:
            traffic_matrix: Traffic matrix to simulate

        Returns:
            dict: Simulation results
        """
        raise NotImplementedError("Booksim simulation is not yet implemented.")

    # ====================================================
    # Workload Aggregate Communication Simulation
    # ====================================================
    def simulate_workload_aggregate_communication(self, retired_networks):
        """
        Simulate aggregate communication for all retired networks.
        
        Args:
            retired_networks (dict): Dictionary of retired mapped networks
            
        Returns:
            dict: Aggregate communication statistics
        """
        # TODO: Update this method to handle phase-based traffic aggregation
        print("\nSimulating aggregate communication...")
        
        # Create a matrix for all traffic
        traffic_matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
        
        # Aggregate traffic from all retired networks
        for network in retired_networks.values():
            for layer_traffic in network.layer_traffic.values():
                if not layer_traffic:
                    continue
                
                for src_chiplet, destinations in layer_traffic.items():
                    for dst_chiplet, amount in destinations.items():
                        traffic_matrix[src_chiplet-1, dst_chiplet-1] += amount
        
        # Run the simulation with "aggregate" type
        # Create a list of tuples (traffic_matrix, network_idx, input_idx, phase_id)
        traffic_matrices = [(traffic_matrix, 0, 0, 0)]  # network_idx=0, input_idx=0, phase_id=0
        return self.simulate_communication(traffic_matrices, "aggregate")

    # ====================================================
    # Network Aggregate Communication Simulation
    # ====================================================
    def simulate_network_aggregate_communication(self, retired_networks):
        """
        Simulate aggregate communication for each individual network.
        
        Args:
            retired_networks (dict): Dictionary of retired mapped networks
            
        Returns:
            dict: Dictionary of simulation results for each network
        """
        # TODO: Update this method to handle phase-based traffic aggregation
        print("\nSimulating network aggregate communication...")
        
        network_results = {}
        
        # Process each network individually
        for network_idx, network in retired_networks.items():
            # Create traffic matrix for this network
            traffic_matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
            
            # Aggregate traffic from all layers in this network
            for layer_traffic in network.layer_traffic.values():
                if not layer_traffic:
                    continue
                
                for src_chiplet, destinations in layer_traffic.items():
                    for dst_chiplet, amount in destinations.items():
                        traffic_matrix[src_chiplet-1, dst_chiplet-1] += amount
            
            # Skip if no traffic
            if not np.any(traffic_matrix):
                continue
            
            # Simulate with network identifier
            simulation_type = f"aggregate"
            # Create a list of tuples (traffic_matrix, network_idx, input_idx, phase_id)
            traffic_matrices = [(traffic_matrix, network_idx, 0, 0)]  # input_idx=0, phase_id=0
            sim_results = self.simulate_communication(traffic_matrices, simulation_type, network_name=network.network_name)
            
            # Store results
            if sim_results:
                network_results[network_idx] = sim_results
        
        return network_results

    # ====================================================
    # Individual Layer Communication Simulation
    # ====================================================
    def simulate_individual_layer_communication(self, retired_networks):
        """
        Simulate communication for each individual layer of each retired network.
        
        Args:
            retired_networks (dict): Dictionary of retired mapped networks
            
        Returns:
            dict: Dictionary of simulation results for each layer
        """
        # TODO: Update this method to simulate individual phase communication instead of layer communication
        print("\nSimulating individual layer communication...")
        
        all_layer_results = {}
        
        # Process each network
        for network_idx, network in retired_networks.items():
            # Process each layer in the network
            for layer_idx, layer_traffic in network.layer_traffic.items():
                # Skip layers without traffic
                if not layer_traffic:
                    continue
                
                # Create traffic matrix for this layer
                traffic_matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets))
                
                # Add layer traffic to matrix
                for src_chiplet, destinations in layer_traffic.items():
                    for dst_chiplet, amount in destinations.items():
                        traffic_matrix[src_chiplet-1, dst_chiplet-1] += amount
                
                # Skip if no traffic
                if not np.any(traffic_matrix):
                    continue
                
                # Simulate with unique identifier
                simulation_type = f"individual_layer"
                # Create a list of tuples (traffic_matrix, network_idx, input_idx, phase_id)
                # TODO: This should use actual phase_id instead of layer_idx
                traffic_matrices = [(traffic_matrix, network_idx, 0, layer_idx)]  # input_idx=0, using layer_idx as phase_id temporarily
                sim_results = self.simulate_communication(traffic_matrices, simulation_type, network_name=network.network_name)
                
                # Store results with network name
                if sim_results:
                    all_layer_results[(network_idx, layer_idx)] = {
                        "results": sim_results,
                        "network_name": network.network_name  # Include network name here
                    }
        
        return all_layer_results

    def _convert_adj_matrix_to_yaml_topology(self, adj_matrix, output_yaml_path):
        """
        Converts a symmetric adjacency matrix from a numpy array to a YAML topology file.
        
        The format of the adjacency matrix should be a square matrix of 0s and 1s.
        The matrix must be symmetric to represent bidirectional connections.
        
        Args:
            adj_matrix (np.ndarray): The adjacency matrix.
            output_yaml_path (str): Path to write the output YAML file.
        
        Raises:
            ValueError: If the adjacency matrix is not symmetric or not square.
        """
        try:
            # Verify that the matrix is square
            if adj_matrix.shape[0] != adj_matrix.shape[1]:
                raise ValueError(f"Adjacency matrix is not square ({adj_matrix.shape[0]}x{adj_matrix.shape[1]}).")

            # Check for symmetry. The matrix must be symmetric for bidirectional connections.
            if not np.array_equal(adj_matrix, adj_matrix.T):
                raise ValueError(f"Adjacency matrix is not symmetric. The topology requires bidirectional connections.")
                
            num_chiplets = adj_matrix.shape[0]
            connections = {}
            
            # Iterate over the matrix to find connections
            for i in range(num_chiplets):
                chiplet_connections = []
                for j in range(num_chiplets):
                    if adj_matrix[i, j] == 1:
                        chiplet_connections.append([i, j])
                
                # Add to dictionary if there are any connections
                if chiplet_connections:
                    connections[f"chiplet_{i}"] = chiplet_connections
            
            # Write the connections to the output YAML file
            with open(output_yaml_path, 'w') as f:
                f.write("connections:\n")
                for chiplet_name, conn_list in connections.items():
                    f.write(f"  {chiplet_name}:\n")
                    for conn in conn_list:
                        f.write(f"    - {str(conn)}\n")
            
            print(f"Successfully converted adjacency matrix to {output_yaml_path}")
            
        except ValueError as ve:
            print(f"Error: {ve}")
            raise
        except Exception as e:
            print(f"An error occurred during YAML conversion: {e}")
            raise