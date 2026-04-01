#!/usr/bin/env python3
# simulation_runner.py

import os
import sys
import time
import pickle
from datetime import datetime
from src.sim import GlobalManager
from src.post.output_manager import OutputManager


class SimulationRunner:
    """
    Handles execution of chiplet system simulations.
    
    This class encapsulates the simulation workflow including initialization,
    execution, and saving of results.
    """
    
    def __init__(self, config):
        """
        Initialize the simulation runner with a configuration.
        
        Args:
            config (dict): Configuration dictionary containing all simulation parameters
        """
        self.config = config
        self.gm = None
        self.raw_results_dir = None
        self.simulation_duration = 0.0
        
    def run(self):
        """
        Execute the simulation and save results.
        
        Returns:
            str: Path to the raw results directory containing simulation outputs
        """
        print("="*80)
        print("🚀 STARTING SIMULATION")
        print("="*80)
        
        start_time = time.time()
        
        # Display configuration
        self._print_config()
        
        # Initialize and run simulation
        self._initialize_global_manager()
        self._run_simulation()
        
        # Save results
        self._create_output_directory()
        self._move_dsent_stats()
        self._save_simulation_state()
        
        # Print summary
        total_duration = time.time() - start_time
        self._print_summary(total_duration)
        
        return self.raw_results_dir
    
    def _print_config(self):
        """Print simulation configuration details."""
        sim_config = self.config['simulation']
        input_files = sim_config['input_files']
        core_settings = sim_config['core_settings']
        
        print(f"⚡ Workload:              {input_files['workload']}")
        print(f"⚡ Communication Sim:     {core_settings['comm_simulator']}")
        print(f"⚡ Communication Method:  {core_settings['comm_method']}")
        print(f"⚡ Adjacency Matrix:      {input_files['adj_matrix']}")
        print(f"⚡ Chiplet Mapping:       {input_files['chiplet_mapping']}")
        print(f"⚡ Model Definitions:     {input_files['model_defs']}")
        print("="*80)
    
    def _initialize_global_manager(self):
        """Initialize the GlobalManager with simulation parameters."""
        print("🔧 Initializing Global Manager...")
        
        sim_config = self.config['simulation']
        input_files = sim_config['input_files']
        core_settings = sim_config['core_settings']
        hw_params = sim_config['hardware_parameters']
        gem5_params = sim_config.get('gem5_parameters', {})
        dsent_params = sim_config.get('dsent_parameters', {})
        
        self.gm = GlobalManager(
            wl_file_name=input_files['workload'],
            adj_matrix_file=input_files['adj_matrix'],
            chiplet_mapping_file=input_files['chiplet_mapping'],
            model_definitions_file=input_files['model_defs'],
            clear_cache=core_settings['clear_cache'],
            communication_simulator=core_settings['comm_simulator'],
            communication_method=core_settings['comm_method'],
            enable_dsent=core_settings['enable_dsent'],
            bits_per_activation=hw_params['bits_per_activation'],
            bits_per_packet=hw_params['bits_per_packet'],
            network_operation_frequency_hz=hw_params['network_operation_frequency_hz'],
            gem5_sim_cycles=gem5_params.get('gem5_sim_cycles', 500000000),
            gem5_injection_rate=gem5_params.get('gem5_injection_rate', 0.0),
            gem5_ticks_per_cycle=gem5_params.get('gem5_ticks_per_cycle', 1000),
            gem5_deadlock_threshold=gem5_params.get('gem5_deadlock_threshold', None),
            dsent_tech_node=dsent_params.get('dsent_tech_node', "32"),
            enable_comm_cache=core_settings['enable_comm_cache'],
            warmup_period_us=core_settings.get('warmup_period_us', 0.0),
            blocking_age_threshold=core_settings['blocking_age_threshold'],
            weight_stationary=core_settings['weight_stationary'],
            weight_loading_strategy=core_settings['weight_loading_strategy'],
        )
        
        print("✅ Global Manager initialized")
    
    def _run_simulation(self):
        """Execute the main simulation."""
        print("\n" + "="*80)
        print("⏳ RUNNING SIMULATION...")
        print("="*80)
        
        sim_start = time.time()
        self.gm.run_simulation()
        self.simulation_duration = time.time() - sim_start
        
        # Store wall clock runtime in the global manager
        self.gm.wall_clock_runtime_s = self.simulation_duration
        
        print("="*80)
        print("✅ SIMULATION COMPLETED")
        print("="*80)
    
    def _create_output_directory(self):
        """Create the output directory for raw results."""
        print("\n📁 Creating output directory...")
        
        output_manager = OutputManager(
            wl_file_name=os.path.basename(self.gm.workload_manager.wl_file),
            adj_matrix_file=os.path.basename(self.gm.adj_matrix_file),
            chiplet_mapping_file=os.path.basename(self.gm.chiplet_mapping_file),
            communication_simulator=self.gm.communication_simulator,
            communication_method=self.gm.communication_method,
            mapping_function=self.gm.mapping_function,
            metric_computer=None,  # Not needed for raw results
            num_chiplets=self.gm.system.num_chiplets,
        )
        
        self.raw_results_dir = output_manager.create_results_directory()
        print(f"✅ Output directory created: {self.raw_results_dir}")
    
    def _save_simulation_state(self):
        """Save the simulation state to a pickle file."""
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        state_filename = f"{timestamp}_post_simulation_system_state.pkl"
        state_filepath = os.path.join(self.raw_results_dir, state_filename)
        
        print(f"\n💾 Saving simulation state to: {state_filepath}...")
        
        try:
            with open(state_filepath, 'wb') as f:
                pickle.dump(self.gm, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ Simulation state saved successfully")
        except Exception as e:
            print(f"❌ ERROR: Failed to save simulation state: {e}")
            sys.exit(1)
    
    def _move_dsent_stats(self):
        """Move DSENT stats file to the results directory if it exists."""
        if not hasattr(self.gm, 'dsent_stats_file'):
            return None
            
        dsent_stats_source = self.gm.dsent_stats_file
        
        if not os.path.exists(dsent_stats_source):
            print("ℹ️  No DSENT stats file found")
            self.gm.dsent_stats_file = None
            return None
            
        if os.path.getsize(dsent_stats_source) == 0:
            print("ℹ️  DSENT stats file is empty")
            self.gm.dsent_stats_file = None
            return None
        
        dsent_stats_filename = os.path.basename(dsent_stats_source)
        dsent_stats_dest = os.path.join(self.raw_results_dir, dsent_stats_filename)
        
        print(f"🚚 Moving DSENT stats file to: {dsent_stats_dest}...")
        
        try:
            os.rename(dsent_stats_source, dsent_stats_dest)
            
            # Clean up temporary directory if empty
            temp_dsent_dir = os.path.dirname(dsent_stats_source)
            if os.path.exists(temp_dsent_dir) and not os.listdir(temp_dsent_dir):
                os.rmdir(temp_dsent_dir)
            
            self.gm.dsent_stats_file = dsent_stats_dest
            if hasattr(self.gm, 'dsent_collector'):
                try:
                    self.gm.dsent_collector.stats_file_path = dsent_stats_dest
                except Exception:
                    pass
            
            print("✅ DSENT stats file moved successfully")
            return dsent_stats_dest
        except Exception as e:
            print(f"⚠️  WARNING: Failed to move DSENT stats file: {e}")
            return None
    
    def _print_summary(self, total_duration):
        """Print simulation summary."""
        print("\n" + "="*80)
        print("🏁 SIMULATION SUMMARY")
        print("="*80)
        print(f"⏱️  Simulation Runtime:      {self.simulation_duration:.2f} seconds")
        print(f"⏱️  Total Script Runtime:    {total_duration:.2f} seconds")
        print(f"📁 Raw Results Directory:   {self.raw_results_dir}")
        print("="*80)
