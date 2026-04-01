#!/usr/bin/env python3
# single_sim_processor.py

import csv
import glob
import os
import pickle
import sys
import time

import numpy as np

from src.post.metrics import MetricComputer, MetricFormatter
from src.post.output_manager import OutputManager
from src.post.simulation_plotter import SimulationPlotter
from src.post.visualize_mapping import ChipletVisualizer


class SingleSimProcessor:
    """
    Processes results from a single simulation run.
    
    This class handles loading simulation results, computing metrics,
    formatting outputs, and generating plots and visualizations.
    """
    
    # Default empty system results for comparison
    DEFAULT_EMPTY_SYSTEM_RESULTS = {
    }
    
    def __init__(self, raw_results_dir, processing_config):
        """
        Initialize the single simulation processor.
        
        Args:
            raw_results_dir (str): Path to the raw results directory
            processing_config (dict): Processing configuration containing:
                - warmup_period_us (float): Warmup period in microseconds
                - run_wkld_agg_comm (bool): Run workload aggregate communication simulation
                - run_ind_comm (bool): Run individual layer communication simulation
                - run_net_agg_comm (bool): Run network aggregate communication simulation
                - generate_plots (bool): Generate plots
                - generate_visualizations (bool): Generate visualizations
        """
        self.raw_results_dir = raw_results_dir
        self.processing_config = processing_config
        
        # Extract processing parameters
        self.warmup_period = processing_config.get('warmup_period_us', 0.0)
        self.cooldown_period = processing_config.get('cooldown_period_us', 0.0)
        self.run_wkld_agg_comm = processing_config.get('run_wkld_agg_comm', False)
        self.run_ind_comm = processing_config.get('run_ind_comm', False)
        self.run_net_agg_comm = processing_config.get('run_net_agg_comm', False)
        self.generate_plots = processing_config.get('generate_plots', True)
        self.generate_visualizations = processing_config.get('generate_visualizations', False)
        
        # Initialize state
        self.gm = None
        self.formatted_results_dir = None
        self.metric_computer = None
        
    def process(self):
        """
        Execute the complete post-processing workflow.
        
        Returns:
            dict: Processing results containing metric_computer and results directory
        """
        print("\n" + "="*80)
        print("📊 STARTING POST-PROCESSING")
        print("="*80)
        print(f"📁 Raw Results Directory: {self.raw_results_dir}")
        
        start_time = time.time()
        
        # Load simulation state
        self._load_simulation_state()
        
        # Apply warmup filtering
        self._apply_warmup_filter()
        
        # Create formatted results directory
        self._create_formatted_results_dir()
        
        # Run supplementary simulations
        self._run_supplementary_simulations()
        
        # Compute metrics
        self._compute_metrics()
        
        # Format and save metrics
        self._format_and_save_metrics()
        
        # Save power data
        self._save_power_data()
        
        # Generate plots
        if self.generate_plots:
            self._generate_plots()
        
        # Generate visualizations
        if self.generate_visualizations:
            self._generate_visualizations()
        
        # Print summary
        duration = time.time() - start_time
        self._print_summary(duration)
        
        return {
            'metric_computer': self.metric_computer,
            'results_dir': self.formatted_results_dir,
            'gm': self.gm
        }
    
    def _load_simulation_state(self):
        """Load the simulation state from the pickle file."""
        print("\n🔄 Loading simulation state...")
        
        # Find the .pkl file that contains the serialized GlobalManager state.
        # Ignore auxiliary pickle files (e.g., dsent_stats.pkl) that now live alongside it.
        pkl_files = [
            path for path in glob.glob(os.path.join(self.raw_results_dir, "*.pkl"))
            if "dsent_stats" not in os.path.basename(path)
        ]
        
        if len(pkl_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 .pkl file in {self.raw_results_dir}, "
                f"but found {len(pkl_files)}: {pkl_files}"
            )
        
        pkl_file = pkl_files[0]
        print(f"📦 Loading from: {pkl_file}")
        
        load_start = time.time()
        try:
            with open(pkl_file, 'rb') as f:
                self.gm = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load simulation state from {pkl_file}: {e}")
        
        load_duration = time.time() - load_start
        print(f"✅ State loaded successfully ({load_duration:.2f}s)")
        
        # Migrate legacy attributes if needed
        if hasattr(self.gm, '_migrate_legacy_attributes'):
            self.gm._migrate_legacy_attributes()
    
    def _apply_warmup_filter(self):
        """Apply warmup period filtering to retired models."""
        print(f"\nℹ️  Applying warmup period: {self.warmup_period} μs")
        
        self.gm.warmup_period_us = self.warmup_period
        # Strictly require Model-based method
        if not hasattr(self.gm, 'filter_retired_models_by_warmup'):
            raise AttributeError("GlobalManager missing required method 'filter_retired_models_by_warmup'")
        self.gm.filter_retired_models_by_warmup()
        
        # Optionally apply cooldown filtering after warmup
        if getattr(self, 'cooldown_period', 0.0) and self.cooldown_period > 0:
            print(f"ℹ️  Applying cooldown period: {self.cooldown_period} μs")
            self.gm.post_warmup_retired_models = self.gm.temporal_filter.filter_by_cooldown(
                retired_models=self.gm.post_warmup_retired_models,
                cooldown_period_us=self.cooldown_period,
                simulation_end_time_us=self.gm.global_time_us
            )
        
        num_models = len(self.gm.post_warmup_retired_models)
        print(f"✅ Warmup filter applied: {num_models} models remaining")
    
    def _create_formatted_results_dir(self):
        """Create the directory for formatted results."""
        print("\n📁 Creating formatted results directory...")
        
        # Get base formatted results directory
        base_formatted_dir = os.path.join(os.getcwd(), "_results", "formatted_results")
        
        # Use the raw results directory name as the formatted directory name
        raw_dir_name = os.path.basename(self.raw_results_dir)
        self.formatted_results_dir = os.path.join(base_formatted_dir, raw_dir_name)
        
        os.makedirs(self.formatted_results_dir, exist_ok=True)
        print(f"✅ Formatted results directory: {self.formatted_results_dir}")
    
    def _run_supplementary_simulations(self):
        """Run supplementary communication simulations if configured."""
        self.workload_aggregate_results = None
        self.individual_results = None
        self.network_aggregate_results = None
        
        if self.run_wkld_agg_comm:
            print("\n🔄 Running workload aggregate communication simulation...")
            start = time.time()
            self.workload_aggregate_results = self.gm.comm_simulator.simulate_workload_aggregate_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Workload aggregate simulation completed ({duration:.2f}s)")
        
        if self.run_ind_comm:
            print("\n🔄 Running individual layer communication simulation...")
            start = time.time()
            self.individual_results = self.gm.comm_simulator.simulate_individual_layer_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Individual layer simulation completed ({duration:.2f}s)")
        
        if self.run_net_agg_comm:
            print("\n🔄 Running network aggregate communication simulation...")
            start = time.time()
            self.network_aggregate_results = self.gm.comm_simulator.simulate_network_aggregate_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Network aggregate simulation completed ({duration:.2f}s)")
    
    def _compute_metrics(self):
        """Compute all metrics from the simulation results."""
        print("\n⚙️  Computing metrics...")
        start = time.time()
        
        # Locate the DSENT stats file. The SimulationRunner now stores the final
        # path on the GlobalManager before serialization, but we keep a small
        # fallback for legacy/raw runs.
        dsent_stats_path = getattr(self.gm, 'dsent_stats_file', None)
        if not dsent_stats_path or not os.path.exists(dsent_stats_path):
            pkl_path = os.path.join(self.raw_results_dir, 'dsent_stats.pkl')
            json_path = os.path.join(self.raw_results_dir, 'dsent_stats.jsonl')
            if os.path.exists(pkl_path):
                dsent_stats_path = pkl_path
            elif os.path.exists(json_path):
                dsent_stats_path = json_path
            else:
                dsent_stats_path = pkl_path  # default path for warning messages
        
        # Initialize metric computer
        self.metric_computer = MetricComputer(
            self.gm.post_warmup_retired_models,
            self.gm.global_time_us,
            self.gm.system.num_chiplets,
            dsent_stats_file_path=dsent_stats_path
        )
        
        # Compute utilization metrics
        self.metric_computer.compute_avg_system_utilization()
        self.metric_computer.compute_utilization_over_time(self.gm.time_step_us)
        
        # Compute model summary metrics
        model_summary_metrics = self.metric_computer.compute_model_summary_metrics()
        print(f"   📈 Generated summary metrics for {len(model_summary_metrics)} models")
        
        # Compute approach comparison metrics
        self.metric_computer.compute_approach_comparison_metrics(
            self.individual_results,
            self.DEFAULT_EMPTY_SYSTEM_RESULTS
        )
        
        # Compute power and energy profiles
        print("   🔋 Computing power profile...")
        self.metric_computer.compute_power_profile(time_step_us=self.gm.time_step_us)
        
        print("   ⚡ Computing energy profile...")
        self.metric_computer.compute_energy_metrics()
        
        duration = time.time() - start
        print(f"✅ Metrics computed successfully ({duration:.2f}s)")
    
    def _format_and_save_metrics(self):
        """Format metrics and save to files."""
        print("\n💾 Formatting and saving metrics...")
        start = time.time()
        
        # Create output manager
        output_manager = OutputManager(
            wl_file_name=os.path.basename(self.gm.workload_manager.wl_file),
            adj_matrix_file=os.path.basename(self.gm.adj_matrix_file),
            chiplet_mapping_file=os.path.basename(self.gm.chiplet_mapping_file),
            communication_simulator=self.gm.communication_simulator,
            communication_method=self.gm.communication_method,
            mapping_function=self.gm.mapping_function,
            metric_computer=self.metric_computer,
            results_dir=self.formatted_results_dir,
            num_chiplets=self.gm.system.num_chiplets
        )
        
        # Create metric formatter
        metric_formatter = MetricFormatter(
            metric_computer=self.metric_computer,
            global_manager=self.gm
        )
        
        # Format all metrics
        formatted_model_metrics = metric_formatter.format_all_model_metrics()
        formatted_utilization_metrics = metric_formatter.format_utilization_metrics(self.gm.time_step_us)
        formatted_comparison_metrics = metric_formatter.format_approach_comparison_metrics()
        
        wall_clock_runtime = self.gm.wall_clock_runtime_s if hasattr(self.gm, 'wall_clock_runtime_s') else 0
        formatted_simulation_summary = metric_formatter.format_simulation_summary(wall_clock_runtime)
        formatted_energy_metrics = metric_formatter.format_energy_metrics()
        
        # Save all formatted metrics
        output_manager.save_formatted_metrics(formatted_model_metrics, subdirectory="formatted_model_metrics")
        output_manager.save_formatted_metrics(formatted_utilization_metrics, subdirectory="formatted_utilization_metrics")
        output_manager.save_formatted_metrics(formatted_comparison_metrics, subdirectory="formatted_comparison_metrics")
        output_manager.save_formatted_metrics(formatted_simulation_summary, subdirectory=None)
        output_manager.save_formatted_metrics(formatted_energy_metrics, subdirectory="formatted_energy_metrics")
        
        duration = time.time() - start
        print(f"✅ Metrics saved successfully ({duration:.2f}s)")
    
    def _save_power_data(self):
        """Save chiplet power data averaged over 100-microsecond intervals."""
        power_data = self.metric_computer.chiplet_total_power_over_time
        
        if not power_data:
            print("\nℹ️  No power data to save")
            return
        
        print("\n💾 Saving 100-microsecond averaged chiplet power data...")
        
        power_csv_path = os.path.join(self.formatted_results_dir, 'chiplet_power_100us_avg.csv')
        time_step_us = self.gm.time_step_us
        
        if time_step_us <= 0:
            print("⚠️  Invalid time step, cannot average power data")
            return
        
        steps_per_100us = int(100 / time_step_us)
        
        if steps_per_100us <= 0:
            print("⚠️  Time step too large to average over 100us intervals")
            return
        
        sorted_chiplet_ids = sorted(power_data.keys())
        max_power = 140.0
        
        with open(power_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for chiplet_id in sorted_chiplet_ids:
                chiplet_name = f"chiplet_chiplet_{chiplet_id}_chiplet"
                power_values = power_data[chiplet_id]
                
                # Pad array to be a multiple of steps_per_100us
                num_values = len(power_values)
                num_to_pad = (steps_per_100us - num_values % steps_per_100us) % steps_per_100us
                
                if num_to_pad > 0:
                    padded_values = np.pad(power_values, (0, num_to_pad), mode='constant', constant_values=np.nan)
                else:
                    padded_values = power_values
                
                # Reshape and compute mean
                reshaped_values = padded_values.reshape(-1, steps_per_100us)
                averaged_power = np.nanmean(reshaped_values, axis=1)
                power_percentages = (averaged_power / max_power) * 100
                
                writer.writerow([chiplet_name] + list(power_percentages))
        
        print(f"✅ Power data saved to: {power_csv_path}")
    
    def _generate_plots(self):
        """Generate plots from the simulation results."""
        print("\n📊 Generating plots...")
        start = time.time()
        
        plotter = SimulationPlotter(
            results_folder=self.formatted_results_dir,
            metric_computer=self.metric_computer
        )
        
        plotter.plot_utilization_over_time()
        plotter.plot_approach_comparison_metrics()
        plotter.plot_power_over_time()
        
        duration = time.time() - start
        print(f"✅ Plots generated successfully ({duration:.2f}s)")
    
    def _generate_visualizations(self):
        """Generate visualizations of network mappings and system state."""
        print("\n🎨 Generating visualizations...")
        start = time.time()
        
        visualizer = ChipletVisualizer(
            adj_matrix_file=self.gm.adj_matrix_file,
            results_folder=self.formatted_results_dir
        )
        
        network_viz_dir = os.path.join(self.formatted_results_dir, "network_mapping_visualizations")
        os.makedirs(network_viz_dir, exist_ok=True)
        
        visualizer.visualize_network_mappings_from_data(
            retired_mapped_models=self.gm.post_warmup_retired_models,
            output_dir=network_viz_dir
        )
        
        visualizer.visualize_system_state_over_time(
            retired_mapped_models=self.gm.post_warmup_retired_models,
            output_dir=network_viz_dir
        )
        
        duration = time.time() - start
        print(f"✅ Visualizations generated successfully ({duration:.2f}s)")
    
    def _print_summary(self, duration):
        """Print processing summary."""
        print("\n" + "="*80)
        print("🏁 POST-PROCESSING SUMMARY")
        print("="*80)
        print(f"⏱️  Processing Time:        {duration:.2f} seconds")
        print(f"📁 Formatted Results:      {self.formatted_results_dir}")
        print(f"📊 Models Processed:       {len(self.gm.post_warmup_retired_models)}")
        print(f"🔋 Plots Generated:        {self.generate_plots}")
        print(f"🎨 Visualizations Created: {self.generate_visualizations}")
        print("="*80)
