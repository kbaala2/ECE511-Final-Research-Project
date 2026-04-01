# global_manager.py

# ====================================================
# A section header
# ====================================================

import os
import sys
import time
from typing import Optional

# Import simulation components
from src.sim.communication_orchestrator import CommunicationOrchestrator
from src.sim.communication_simulator import CommunicationSimulator
from src.sim.compute_simulation import ComputeSimulator
from src.sim.scheduling import WeightScheduler

# Import core components
from src.core.system import System
from src.core.mapped_model import MappedModel
from src.core.traffic_calculator import TrafficCalculator
from src.core.model_processor import ModelProcessor

# Import managers
from src.managers.workload_manager import WorkloadManager
from src.managers.model_def_manager import ModelDefinitionManager

# Import mapping components
from src.mapping.model_mapper import ModelMapper
from src.mapping.layer_partitioner import LayerPartitioner

# Import utility components
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.stats_collector import StatsCollector

# Import post-simulation components
from src.post.temporal_filter import TemporalFilter

# GlobalManager class
class GlobalManager:
    """
    Main class responsible for managing the chiplet simulation, including workload processing,
    model mapping, computation, and communication.
    
    Args:
        wl_file_name (str): Name of the workload file, defaults to "workload_1_0ms.csv"
        adj_matrix_file (str): Name of the adjacency matrix file, defaults to "adj_matrix_86x86.csv"
        chiplet_mapping_file (str): Name of the chiplet mapping file, defaults to "chiplet_mapping_86.yaml"
        model_definitions_file (str): Path to model definitions file, defaults to "model_definitions.py"
        compute_cache_file (str): Name of the compute cache file, defaults to "compute_cache.pkl"
        communication_cache_file (str): Name of the communication cache file, defaults to "communication_cache.pkl"
        clear_cache (bool): Whether to clear the cache, defaults to False
        communication_simulator (str): Simulator to use for communication ("Garnet" or "Booksim"), defaults to "Garnet"
        communication_method (str): Method for communication ("pipelined" or "non-pipelined"), defaults to "non-pipelined"
        warmup_period_us (float): Warmup period in microseconds, defaults to 0.0
        enable_dsent (bool): Whether to enable DSENT simulation, defaults to True
        weight_stationary (bool): Whether weights are stationary (loaded once per model), defaults to True
        weight_loading_strategy (str): Weight loading strategy ("all_at_once" or "just_in_time"), defaults to "all_at_once"
    """
    def __init__(
        self,
        wl_file_name = "workload_1_0ms.csv",   # Name of the workload file
        adj_matrix_file = "adj_matrix_10x10_mesh_with_io.csv",  # Now using topology with I/O chiplets
        chiplet_mapping_file = "mapping_102_with_io.yaml",  # Now using mapping with I/O chiplets
        model_definitions_file = "model_definitions.py",  # Path to model definitions file
        compute_cache_file = "compute_cache.pkl",
        communication_cache_file = "communication_cache.pkl",  # Add parameter for communication cache
        clear_cache = False,
        communication_simulator = "Garnet",
        communication_method = "non-pipelined",  # "pipelined" or "non-pipelined"
        warmup_period_us = 0.0,
        enable_dsent=True,
        blocking_age_threshold=10,
        # Parameterized communication settings
        bits_per_activation: int = 8,
        bits_per_packet: int = 128,
        network_operation_frequency_hz: int = 1_000_000_000,
        gem5_sim_cycles: int = 500_000_000,
        gem5_injection_rate: float = 0.0,
        gem5_ticks_per_cycle: int = 1000,
        gem5_deadlock_threshold: Optional[int] = None,
        dsent_tech_node: str = "32",
        enable_comm_cache: bool = False,
        weight_stationary: bool = True,  # Weight stationary mode for in-memory computing
        weight_loading_strategy: str = "all_at_once",  # Weight loading strategy: "all_at_once" or "just_in_time"
        ):
        """Initialize the GlobalManager by setting up all components in organized stages."""
        
        # Get the project root directory (2 levels up from src/sim)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Stage 1: Initialize configuration and parameters
        self._initialize_configuration(
            communication_method=communication_method,
            communication_simulator=communication_simulator,
            weight_stationary=weight_stationary,
            weight_loading_strategy=weight_loading_strategy,
            warmup_period_us=warmup_period_us,
            clear_cache=clear_cache,
        )
        
        # Stage 2: Setup all file paths
        self._setup_paths(
            project_root=project_root,
            chiplet_mapping_file=chiplet_mapping_file,
            adj_matrix_file=adj_matrix_file,
            compute_cache_file=compute_cache_file,
            communication_cache_file=communication_cache_file,
            model_definitions_file=model_definitions_file,
        )
        
        # Stage 3: Initialize managers
        self._initialize_managers(
            project_root=project_root,
            wl_file_name=wl_file_name,
            blocking_age_threshold=blocking_age_threshold,
        )
        
        # Stage 4: Initialize system components
        self._initialize_system_components(
            bits_per_activation=bits_per_activation,
            bits_per_packet=bits_per_packet,
        )
        
        # Stage 5: Store communication parameters
        self._store_communication_parameters(
            network_operation_frequency_hz=network_operation_frequency_hz,
            bits_per_activation=bits_per_activation,
            bits_per_packet=bits_per_packet,
            gem5_sim_cycles=gem5_sim_cycles,
            gem5_injection_rate=gem5_injection_rate,
            gem5_ticks_per_cycle=gem5_ticks_per_cycle,
            gem5_deadlock_threshold=gem5_deadlock_threshold,
            dsent_tech_node=dsent_tech_node,
            enable_comm_cache=enable_comm_cache,
        )
        
        # Stage 6: Initialize simulators
        self._initialize_simulators(
            enable_dsent=enable_dsent,
        )
        
        # Stage 7: Setup monitoring and statistics collection
        self._setup_monitoring_and_stats(
            project_root=project_root,
            enable_dsent=enable_dsent,
        )
        
        # Stage 8: Initialize simulation state tracking
        self._initialize_simulation_state()
        
        # Stage 9: Initialize orchestration components
        self._initialize_orchestration()
    
    def _initialize_configuration(
        self,
        communication_method: str,
        communication_simulator: str,
        weight_stationary: bool,
        weight_loading_strategy: str,
        warmup_period_us: float,
        clear_cache: bool,
    ):
        """Initialize and validate configuration parameters."""
        # Communication settings
        self.communication_method = communication_method
        self.communication_simulator = communication_simulator
        
        # Weight management settings
        self.weight_stationary = weight_stationary
        self.weight_loading_strategy = weight_loading_strategy
        
        # Validate weight loading strategy
        if weight_loading_strategy not in ["all_at_once", "just_in_time"]:
            raise ValueError(
                f"Invalid weight_loading_strategy: {weight_loading_strategy}. "
                "Must be 'all_at_once' or 'just_in_time'"
            )
        
        # Simulation settings
        self.mapping_function = "nearest_neighbor_v3"
        self.warmup_period_us = warmup_period_us
        self.time_step_us = 1  # In microseconds
        self.clear_cache = clear_cache
        
        # Initialize energy tracking
        self.chiplet_compute_energy = {}
    
    def _setup_paths(
        self,
        project_root: str,
        chiplet_mapping_file: str,
        adj_matrix_file: str,
        compute_cache_file: str,
        communication_cache_file: str,
        model_definitions_file: str,
    ):
        """Setup all file paths used by the simulator."""
        # Setup asset paths
        self.chiplet_mapping_file = os.path.join(project_root, "assets", "chiplet_specs", chiplet_mapping_file)
        self.adj_matrix_file = os.path.join(project_root, "assets", "NoI_topologies", adj_matrix_file)
        
        # Setup cache paths
        self.compute_cache_file = os.path.join(project_root, "cache", compute_cache_file)
        self.communication_cache_file = os.path.join(project_root, "cache", communication_cache_file)
        
        # Setup model definitions path
        self.model_definitions_file = os.path.join(project_root, "assets", "DNN_models", model_definitions_file)
    
    def _initialize_managers(
        self,
        project_root: str,
        wl_file_name: str,
        blocking_age_threshold: int,
    ):
        """Initialize workload and model definition managers."""
        # Initialize WorkloadManager
        wl_file_path = os.path.join(project_root, "assets", "workloads", wl_file_name)
        self.workload_manager = WorkloadManager(
            wl_file_path,
            blocking_age_threshold=blocking_age_threshold
        )
        
        # Initialize ModelDefinitionManager
        self.model_manager = ModelDefinitionManager(self.model_definitions_file)
        self.models = self.model_manager.models
    
    def _initialize_system_components(
        self,
        bits_per_activation: int,
        bits_per_packet: int,
    ):
        """Initialize system, mapper, partitioner, and traffic calculator."""
        # Initialize the System
        self.system = System(
            chiplet_mapping_file=self.chiplet_mapping_file,
            adj_matrix_file=self.adj_matrix_file
        )
        
        # Initialize the ModelMapper
        self.model_mapper = ModelMapper(
            system=self.system,
            mapping_function=self.mapping_function
        )
        # Optional seeding map: {model_idx: chiplet_id}
        self.seed_start_chiplet_ids = {}
        # Optional allowed chiplets per model: {model_idx: Iterable[int]}
        self.allowed_chiplet_ids = {}
        
        # Initialize the LayerPartitioner
        self.layer_partitioner = LayerPartitioner()
        
        # Initialize the TrafficCalculator
        self.traffic_calculator = TrafficCalculator(
            system=self.system,
            bits_per_activation=bits_per_activation,
            bits_per_packet=bits_per_packet
        )
    
    def _store_communication_parameters(
        self,
        network_operation_frequency_hz: int,
        bits_per_activation: int,
        bits_per_packet: int,
        gem5_sim_cycles: int,
        gem5_injection_rate: float,
        gem5_ticks_per_cycle: int,
        gem5_deadlock_threshold: Optional[int],
        dsent_tech_node: str,
        enable_comm_cache: bool,
    ):
        """Store communication-related parameters."""
        self.network_operation_frequency = network_operation_frequency_hz
        self.bits_per_activation = bits_per_activation
        self.bits_per_packet = bits_per_packet
        self.gem5_sim_cycles = gem5_sim_cycles
        self.gem5_injection_rate = gem5_injection_rate
        self.gem5_ticks_per_cycle = gem5_ticks_per_cycle
        self.gem5_deadlock_threshold = gem5_deadlock_threshold
        self.dsent_tech_node = dsent_tech_node
        self.enable_comm_cache = enable_comm_cache
        # Allow test override via environment variable SIM_ENABLE_COMM_CACHE
        env_cache = os.getenv('SIM_ENABLE_COMM_CACHE', '').lower()
        if env_cache in ('1', 'true', 'yes', 'on'):
            self.enable_comm_cache = True
    
    def _initialize_simulators(self, enable_dsent: bool):
        """Initialize compute and communication simulators."""
        # Initialize ComputeSimulator
        self.compute_simulator = ComputeSimulator(
            cache_file=self.compute_cache_file,
            clear_cache=self.clear_cache,
        )
        
        # Initialize CommunicationSimulator
        self.comm_simulator = CommunicationSimulator(
            system=self.system,
            simulator_type=self.communication_simulator,
            communication_method=self.communication_method,
            cache_file=self.communication_cache_file,
            clear_cache=self.clear_cache,
            network_operation_frequency=self.network_operation_frequency,
            enable_dsent=enable_dsent,
            enable_cache=self.enable_comm_cache,
            dsent_tech_node=self.dsent_tech_node,
            gem5_sim_cycles=self.gem5_sim_cycles,
            gem5_injection_rate=self.gem5_injection_rate,
            gem5_ticks_per_cycle=self.gem5_ticks_per_cycle,
            gem5_deadlock_threshold=self.gem5_deadlock_threshold,
            gem5_topology_config_relpath="configs/topologies/myTopology.yaml",
        )
        
        # Log DSENT configuration status
        if enable_dsent:
            print(f"✅ DSENT enabled with tech node: {self.dsent_tech_node}")
        else:
            print("⚠️ DSENT disabled - no power/energy/area stats will be collected")
    
    def _setup_monitoring_and_stats(self, project_root: str, enable_dsent: bool):
        """Setup performance monitoring and statistics collection."""
        # Initialize performance monitoring
        timing_dir = os.path.join(project_root, "temp", "timing")
        os.makedirs(timing_dir, exist_ok=True)
        self.timing_log_file = os.path.join(timing_dir, "timing_summary.log")
        self.performance_monitor = PerformanceMonitor(log_file_path=self.timing_log_file)
        
        # Create convenience references to performance monitor methods
        self.timing_stats = self.performance_monitor.timing_stats
        self._time_operation = self.performance_monitor.time_operation
        
        # Setup DSENT statistics collector
        temp_dir = os.path.join(os.getcwd(), "temp")
        dsent_stats_dir = os.path.join(temp_dir, "dsent")
        os.makedirs(dsent_stats_dir, exist_ok=True)
        
        # Clean existing DSENT stats files
        if os.path.exists(dsent_stats_dir):
            for filename in os.listdir(dsent_stats_dir):
                file_path = os.path.join(dsent_stats_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        dsent_stats_file = os.path.join(dsent_stats_dir, "dsent_stats.pkl")
        self.dsent_collector = StatsCollector(
            stats_file_path=dsent_stats_file,
            dump_threshold=1,
            stats_type="dsent",
            auto_initialize_file=True,
            serialization_format="pickle"
        )
        self.dsent_stats_file = dsent_stats_file
        self.dsent_stats_format = "pickle"
        
        # Store DSENT flag for later use
        self.enable_dsent_flag = enable_dsent
    
    def _initialize_simulation_state(self):
        """Initialize all simulation state tracking variables."""
        # Model tracking
        self.active_mapped_models = {}
        self.retired_mapped_models = {}
        self.post_warmup_retired_models = {}
        
        # Time tracking
        self.global_time_us = 0.0
        self.wall_clock_runtime_s = 0.0
        
        # Counters
        self.simulation_call_counter = 0
    
    def _initialize_orchestration(self):
        """Initialize orchestration components (processors, schedulers, orchestrators)."""
        # Initialize model processor
        self.model_processor = ModelProcessor()
        
        # Initialize weight scheduler
        self.weight_scheduler = WeightScheduler(strategy=self.weight_loading_strategy)
        
        # Initialize communication orchestrator
        self.comm_orchestrator = CommunicationOrchestrator(
            comm_simulator=self.comm_simulator,
            dsent_collector=self.dsent_collector,
            system=self.system
        )
        
        # Initialize temporal filter for warmup/cooldown filtering
        self.temporal_filter = TemporalFilter(
            print_header_func=self._print_header
        )
    
    
    def _print_header(self, title, char="═", box_width=53):
        """
        Helper method to print formatted headers consistently
        
        Args:
            title (str): The title to display
            char (str): Character to use for the border
            box_width (int): Width of the box
        """
        print("\n" + "=" * 80)
        print(f"╔{char * (box_width)}╗")
        print(f"║{title.center(box_width)}║")
        print(f"╚{char * (box_width)}╝")
    
    def simulate_communication(self):
        """
        Simulate communication for all active models during the current timestep.
        
        Delegates to CommunicationOrchestrator for the actual simulation.
        """
        return self.comm_orchestrator.simulate_communication(
            self.active_mapped_models,
            self.global_time_us
        )

    def simulate_compute(self, model_name, layer_idx, chiplet_id, partitioned_layer, num_layers, model_idx):
        """
        Simulate the compute of a single model chunk.
        Automatically selects backend based on chiplet type (IMC vs CMOS).
        
        Args:
            model_name (str): Name of the model (e.g., 'ResNet18').
            layer_idx (int): Index of the layer in the model.
            chiplet_id (int): ID of the chiplet to simulate on.
            partitioned_layer (dict): Partitioned layer definition.
            num_layers (int): Total number of layers in the model.
            model_idx (int): Index of the model in the workload.
            
        Returns:
            dict: Simulation results containing latency, energy, etc.
        """
        
        # Print information that compute is starting
        print(f"🔹 Compute simulation is starting for model {model_idx} ({model_name}), layer {layer_idx}, chiplet {chiplet_id}")
        
        # Get the chiplet type and compute type
        chiplet_type = self.system.chiplet_mapping[chiplet_id]
        compute_type = self.system.get_chiplet_compute_type(chiplet_id)
        chiplet_params = self.system.get_chiplet_params(chiplet_id)
        
        # Log the compute type being used
        print(f"📊 Chiplet {chiplet_id} type: {chiplet_type}, compute type: {compute_type}")
        
        # Use the compute simulator to run the simulation
        return self.compute_simulator.simulate_compute(
            model_name=model_name,
            layer_idx=layer_idx,
            chiplet_id=chiplet_id,
            partitioned_layer=partitioned_layer,
            num_layers=num_layers,
            model_idx=model_idx,
            chiplet_type=chiplet_type,
            compute_type=compute_type,
            chiplet_params=chiplet_params
        )

    def run_simulation(self):
        print("╔═════════════════════════════════════════════════════╗")
        print("║             STARTING SIMULATION RUN                 ║")
        print("╚═════════════════════════════════════════════════════╝")
        
        # Check if workload is empty
        if self.workload_manager.get_remaining_count() == 0:
            print("❌ ERROR: Workload file is empty or could not be read.")
            sys.exit(1)
        
        # Track overall simulation time
        simulation_start_time = time.time()
        
        # Reset global time tracker
        self.global_time_us = 0.0
        
        print(f"\n📋 Simulation will process {self.workload_manager.get_remaining_count()} model entries from the workload")
        
        # ====================================================
        # While there are models to process, loop while incrementing the global time
        # ====================================================
        loop_iteration = 0
        while self.workload_manager.get_remaining_count() > 0 or self.active_mapped_models:
            loop_start_time = time.time()
            print(f"\n⏱️ Global Time: {self.global_time_us:.2f} μs (Loop {loop_iteration})")
            print(f"🔄 Remaining model entries to process: {self.workload_manager.get_remaining_count()}")
            print(f"📊 Active models: {len(self.active_mapped_models)}")

            # Get models that need to be injected at the current time step
            print("🔍 Getting models to inject...")
            with self._time_operation('get_inject_models'):
                models_to_inject = self.workload_manager.get_models_to_inject(self.global_time_us)
            #self.timing_logger.info(f"[Loop {loop_iteration} - Time {self.global_time_us:.2f} us] Get Inject Models took: {get_inject_duration:.6f} s")

            # Process models to inject at this time step (if any)
            inject_loop_start = time.time()
            models_processed_this_step = 0 # Track successful injections in this step
            if not models_to_inject:
                # No models to inject at current time - silent behavior
                print("ℹ️ No models to inject at this time step")
            else:
                print(f"📥 Processing {len(models_to_inject)} models to inject...")
                # ====================================================
                # Loop over each model that needs to be injected at the current time
                # ====================================================
                for model_name, num_inputs, model_idx in models_to_inject:
                    model_process_start = time.time()

                    # Get the model definition
                    if model_name not in self.models:
                        print(f"❌ ERROR: Model '{model_name}' not found in available models")
                        sys.exit(1)
                    model_def = self.models[model_name]

                    # Process the model and collect metrics using ModelProcessor
                    model_metrics = self.model_processor.process_model(model_def)

                    # First, check if the model can ever fit in the system (definitive capacity check)
                    can_ever_fit, fit_check_reason = self.system.can_model_ever_fit(model_metrics)
                    if not can_ever_fit:
                        # Model is too large for the entire system - this is a fatal error
                        total_system_memory = self.system.get_total_system_memory()
                        model_memory_requirement = self.system.get_model_memory_requirement(model_metrics)
                        
                        print("\n" + "="*80)
                        print("❌ FATAL ERROR: MODEL TOO LARGE FOR SYSTEM")
                        print("="*80)
                        print(f"Model: {model_name} (ID: {model_idx})")
                        print(f"Number of layers: {len(model_metrics)}")
                        print(f"Model memory requirement: {model_memory_requirement:,.0f} weights")
                        print(f"Total system memory: {total_system_memory:,.0f} weights")
                        print(f"Memory deficit: {model_memory_requirement - total_system_memory:,.0f} weights")
                        if total_system_memory > 0:
                            print(f"Memory utilization: {(model_memory_requirement / total_system_memory * 100):.1f}% of system capacity")
                        else:
                            print("Memory utilization: N/A (total system memory reported as 0)")
                        print(f"System has {self.system.num_chiplets} chiplets")
                        print(f"Compute chiplets: {self.system.num_chiplets - len(self.system.io_chiplet_ids)}")
                        print(f"I/O chiplets: {len(self.system.io_chiplet_ids)}")
                        print("="*80)
                        print("This model will never fit in the system regardless of current memory usage.")
                        print("Consider using a smaller model or a system with more memory capacity.")
                        print("Simulation terminated.")
                        print("="*80)
                        sys.exit(1)

                    # Second, check if the model can fit now with currently available memory
                    can_map, pre_check_reason = self.system.can_model_fit_now(model_metrics)
                    if not can_map:
                        self.workload_manager.increment_failure_age(model_idx)
                        print(f"⚠️ Model {model_idx} ({model_name}) pre-check failed ({pre_check_reason}). Incrementing failure age.")
                        if self.workload_manager.is_model_blocking(model_idx):
                            print(f"🛑 Model {model_idx} is now blocking. Advancing simulation time.")
                            break
                        continue

                    # Generate the mapping for the model
                    with self._time_operation('mapping_generation'):
                        # Build mapping preference for this model (seed + allowed chiplets if provided)
                        mapper_preference = {}
                        if model_idx in self.seed_start_chiplet_ids:
                            mapper_preference["preferred_start_chiplet_id"] = self.seed_start_chiplet_ids[model_idx]
                        if model_idx in self.allowed_chiplet_ids:
                            mapper_preference["allowed_chiplet_ids"] = list(self.allowed_chiplet_ids[model_idx])

                        if mapper_preference:
                            prev_pref = self.model_mapper.preference
                            try:
                                if prev_pref:
                                    # Merge, with mapper_preference taking precedence
                                    merged = {**prev_pref, **mapper_preference}
                                else:
                                    merged = mapper_preference
                                self.model_mapper.preference = merged
                                mapping, used_capacity, failure_reason = self.model_mapper.generate_mapping(model_metrics=model_metrics)
                            finally:
                                self.model_mapper.preference = prev_pref
                        else:
                            mapping, used_capacity, failure_reason = self.model_mapper.generate_mapping(model_metrics=model_metrics)
                        
                    if mapping is None:
                        if failure_reason == "INSUFFICIENT_MEMORY" or failure_reason == "NO_AVAILABLE_CHIPLETS":
                            # New logic: increment failure age and continue to the next model
                            self.workload_manager.increment_failure_age(model_idx)
                            print(f"⚠️ Model {model_idx} ({model_name}) mapping failed ({failure_reason}). Incrementing failure age.")
                            
                            # If the failed model is now blocking, stop trying to map more models this timestep
                            if self.workload_manager.is_model_blocking(model_idx):
                                print(f"🛑 Model {model_idx} is now blocking. Advancing simulation time.")
                                break # Stop processing more models at this time step
                                
                            continue # Try the next model in the prioritized list
                        else:
                            print(f"❌ Failed to generate mapping for model: {model_name}, reason: {failure_reason}")
                            #self.timing_logger.error(f"  [Net {model_idx} ({model_name})] Failed to generate mapping: {failure_reason}")
                            sys.exit(1)

                    # Mark the model as successfully injected since mapping succeeded
                    self.workload_manager.mark_model_injected(model_idx)
                    models_processed_this_step += 1

                    print(f"✅ Model {model_idx} ({model_name}) successfully injected at time {self.global_time_us:.2f} μs")

                    # After the mapping is generated, create a MappedModel object to store the mapping and other metrics
                    mapped_model = MappedModel(
                        model_name=model_name,
                        mapping=mapping,
                        model_start_time_us=self.global_time_us,
                        model_def=model_def,
                        num_inputs=num_inputs,
                        used_capacity=used_capacity,
                        model_metrics=model_metrics,
                        model_idx=model_idx,
                        communication_method=self.communication_method,
                        traffic_calculator=self.traffic_calculator,
                        weight_stationary=self.weight_stationary,
                    )

                    # Add the MappedModel to the dictionary of active MappedModels using model index as key
                    self.active_mapped_models[model_idx] = mapped_model
                    
                    # Generate weight loading schedule based on configured strategy
                    weight_loading_schedule = self.weight_scheduler.generate_schedule(mapped_model.num_layers)
                    mapped_model.add_weight_loading_phases(weight_loading_schedule)
                    
                    # Configure dependencies for the selected strategy (all_at_once or just_in_time)
                    self.weight_scheduler.configure_dependencies(mapped_model)
                    
                    # Print schedule summary for debugging
                    if self.weight_loading_strategy == "just_in_time":
                        self.weight_scheduler.print_schedule_summary(mapped_model, model_idx)
                
                    # Count unique chiplets used in the mapping
                    unique_chiplets = set()
                    for layer_idx, chiplet_mappings in mapping:
                        for chiplet_id, percentage in chiplet_mappings:
                            unique_chiplets.add(chiplet_id)
                    
                    print(f"🎯 Mapping completed for model {model_idx} ({model_name}). Using {len(unique_chiplets)} chiplets: {sorted(unique_chiplets)}")

                    # ====================================================
                    # Loop over each layer in the model, partition the layer, and run compute simulation
                    # ====================================================
                    for layer_idx, chiplet_mappings in mapping:
                        print(f"\n🔹 Layer {layer_idx}: {len(chiplet_mappings)} chiplet assignments")

                        # Determine layer definition using the new layer format
                        if 'layers' in model_def and layer_idx in model_def['layers']:
                            layer_def = model_def['layers'][layer_idx].copy()  # Make a copy to avoid modifying original
                        else:
                            print(f"❌ Invalid layer index {layer_idx} for model '{model_name}' (new format)")
                            sys.exit(1)

                        # Add total_macs from model_metrics to layer_def for accurate compute simulation
                        if layer_idx < len(model_metrics):
                            layer_def['total_macs'] = model_metrics[layer_idx]['total_macs']
                            print(f"🔍 Added total_macs {model_metrics[layer_idx]['total_macs']:,.0f} to layer {layer_idx}")

                        # Partition the layer
                        with self._time_operation('layer_partitioning'):
                            layer_chunks = self.layer_partitioner.partition_layer(layer_def=layer_def, chiplet_mappings=chiplet_mappings)
                        #self.timing_logger.info(f"    [Net {model_idx}, Layer {layer_idx}] Partitioning took: {partition_duration:.6f} s")
                        print(f"📊 Layer partitioned into {len(layer_chunks)} chunks")

                        # Save the layer chunks to the MappedModel object
                        mapped_model.set_layer_chunks(layer_idx, layer_chunks)

                        # Process each chunk of the layer and run compute simulation(assigned to different chiplets)
                        #compute_loop_start = time.time() # Covered by layer processing within model_process_start
                        for chunk_idx, (chiplet_id, partitioned_layer) in enumerate(layer_chunks):
                            # Add layer index information to the partitioned layer
                            partitioned_layer['layer_idx'] = layer_idx

                            print(f"\n⚙️  Simulating compute chunk for chiplet {chiplet_id}...")

                            # Simulate this specific chunk using the updated method
                            with self._time_operation('compute_chunk'):
                                chunk_result = self.simulate_compute(
                                    model_name=model_name,
                                    layer_idx=layer_idx,
                                    chiplet_id=chiplet_id,
                                    partitioned_layer=partitioned_layer,
                                    num_layers=mapped_model.num_layers,
                                    model_idx=model_idx
                                )

                            # Add the chunk result to the MappedModel object with chunk index
                            mapped_model.add_chunk_compute_result(layer_idx, chiplet_id, chunk_idx, chunk_result)
                    # ====================================================
                    # Compute simulation for each layer in the model complete, save results and calculate traffic
                    # ====================================================

                    # Update all layer compute metrics for the model
                    with self._time_operation('update_compute_metrics'):
                        mapped_model.update_all_layer_compute_metrics()

                    # Calculate traffic including weight loading phases
                    with self._time_operation('calculate_traffic'):
                        mapped_model.calculate_all_traffic()
                    
                    # After traffic has been calculated, mark the layers without communication
                    with self._time_operation('mark_no_communication'):
                        mapped_model.mark_layers_without_communication()

                    model_process_duration = time.time() - model_process_start # Recalculate here to cover everything for this model
                    if 'total_model_processing' not in self.timing_stats:
                        self.timing_stats['total_model_processing'] = {'total_time': 0.0, 'count': 0}
                    self.timing_stats['total_model_processing']['total_time'] += model_process_duration
                    self.timing_stats['total_model_processing']['count'] += 1
                    print(f"\n⏱️  Model {model_name} processing took {model_process_duration:.3f} seconds")

            inject_loop_duration = time.time() - inject_loop_start
            if models_to_inject: # Log only if injection loop ran
                if 'process_injected_models_loop' not in self.timing_stats:
                    self.timing_stats['process_injected_models_loop'] = {'total_time': 0.0, 'count': 0}
                self.timing_stats['process_injected_models_loop']['total_time'] += inject_loop_duration
                self.timing_stats['process_injected_models_loop']['count'] += 1 # Count loops where injection happened

            # ====================================================
            # All models have been injected for the current time step, simulate communication
            # ====================================================

            # Call the simulate_communication function to simulate communication
            print("🌐 Simulating communication...")
            with self._time_operation('simulate_communication'):
                self.simulate_communication()

            # The state of system crossbars occupied has already been increased by the mapping operation, so no need to do it here
            pass
            
            # ====================================================
            # Recalculate completion times for all active models after communication updates
            # ====================================================
            print("⏰ Recalculating completion times...")
            for model_idx in self.active_mapped_models.keys():
                mapped_model = self.active_mapped_models[model_idx]
                # This will update completion times based on any latency changes
                mapped_model.calculate_phase_timing(self.global_time_us)

            # ====================================================
            # Free up crossbars occupied by retired models
            # ====================================================
            print("🔄 Checking for retired models...")
            with self._time_operation('retire_check'):
                retired_count = 0
                for model_idx in list(self.active_mapped_models.keys()):
                    mapped_model = self.active_mapped_models[model_idx]

                    # Update the completion times for the model
                    # This may update completion times or next input start times
                    completion_updated = mapped_model.calculate_phase_timing(self.global_time_us)

                    # Check if the model is complete *after* attempting to update times
                    is_complete = mapped_model.is_complete(self.global_time_us)
                    
                    if is_complete:
                        # TODO: Add error checking to ensure that the model has all the correct attributes
                        self.retired_mapped_models[model_idx] = mapped_model
                        del self.active_mapped_models[model_idx]
                        retired_count += 1

                        # Free up the crossbars occupied by the model.

                        # Capacity-aware free: IMC uses crossbars, CMOS uses weight units
                        prev_available_capacity = self.system.get_available_capacity_per_chiplet()
                        new_available_capacity = prev_available_capacity + mapped_model.used_capacity
                        self.system.update_capacity_availability(new_available_capacity)
                        print(f"    ✅ Model {model_idx} retired")
                
                print(f"  Retirement check complete: {retired_count} models retired, {len(self.active_mapped_models)} still active")

            # Advance global time by one time step
            print("⏭️ Advancing time...")
            self.global_time_us += self.time_step_us
            # Timing stats
            loop_iteration += 1
            # Track total loop time
            loop_duration = time.time() - loop_start_time
            if 'total_loop_iteration' not in self.timing_stats:
                self.timing_stats['total_loop_iteration'] = {'total_time': 0.0, 'count': 0}
            self.timing_stats['total_loop_iteration']['total_time'] += loop_duration
            self.timing_stats['total_loop_iteration']['count'] += 1
            print(f"✅ Loop {loop_iteration} completed in {loop_duration:.3f}s")


        # ====================================================
        # Simulation Complete, Print results
        # ====================================================
        # Calculate total simulation time
        simulation_duration = time.time() - simulation_start_time
        print("\n" + "=" * 80 + "\n")
        self._print_header("SIMULATION COMPLETE", box_width=63)
        print(f"⏱️  Total simulation run took {simulation_duration:.3f} seconds")

        # ====================================================
        # Log Aggregate Timing Summary
        # ====================================================
        summary_header = "\nTiming Summary Report\n" + "-"*30
        print(summary_header)
        with open(self.timing_log_file, 'a') as f:
            f.write(f"Overall Simulation Duration: {simulation_duration:.6f} seconds\n")
            f.write(summary_header + "\n")
            f.write(f"{'Section':<35} | {'Total Time (s)':<15} | {'Count':<10} | {'Avg Time (s)':<15}\n")
            f.write("-"*80 + "\n")

            for section, stats in self.timing_stats.items():
                total_time = stats['total_time']
                count = stats['count']
                avg_time = total_time / count if count > 0 else 0
                summary_line = f"{section:<35} | {total_time:<15.6f} | {count:<10} | {avg_time:<15.6f}"
                print(summary_line)
                f.write(summary_line + "\n")

            f.write("="*80 + "\n")
            print("Timing summary logged to", self.timing_log_file)
            
        # Save the detailed mapper timing statistics
        if hasattr(self.model_mapper, 'save_timing_stats'):
            self.model_mapper.save_timing_stats()
            print("\nDetailed mapper timing statistics logged to", self.model_mapper.mapping_log_file)

        # Dump any remaining DSENT stats at the end of the simulation
        self.dsent_collector.dump_stats()
    
    def filter_retired_models_by_warmup(self):
        """
        Filters the retired models, keeping only those that started
        at or after the specified warmup period.
        Stores the result in self.post_warmup_retired_models.
        
        Delegates to TemporalFilter.
        """
        self.post_warmup_retired_models = self.temporal_filter.filter_by_warmup(
            self.retired_mapped_models,
            self.warmup_period_us
        )
                    
