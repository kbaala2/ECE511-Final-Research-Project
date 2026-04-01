import glob
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap


class CrossSimProcessor:
    """
    Standalone cross-simulation analysis processor.
    
    Loads results from multiple simulation runs and performs comparative analysis.
    """
    
    def __init__(self, result_directories, base_output_dir=None, analysis_name="cross_simulation_analysis"):
        """
        Initialize the cross-simulation processor.
        
        Args:
            result_directories (list): List of raw results directory paths to analyze
            base_output_dir (str, optional): Base directory for output. Defaults to _results/
            analysis_name (str, optional): Name for this analysis. Defaults to "cross_simulation_analysis"
        """
        self.result_directories = result_directories
        
        # Set base output directory
        if base_output_dir is None:
            self.base_output_dir = os.path.join(os.getcwd(), "_results")
        else:
            self.base_output_dir = base_output_dir
            
        self.analysis_name = analysis_name
        self.cross_analysis_output_dir = os.path.join(
            self.base_output_dir, 
            "cross_simulation_analysis_results",
            analysis_name
        )
        
        # Loaded data from result directories
        self.collected_analysis_data = []
        
        # Create organized subdirectories for different plot categories
        self.performance_dir = os.path.join(self.cross_analysis_output_dir, "performance_analysis")
        self.execution_time_dir = os.path.join(self.cross_analysis_output_dir, "execution_time_analysis")
        self.utilization_dir = os.path.join(self.cross_analysis_output_dir, "utilization_analysis")
        self.comparison_dir = os.path.join(self.cross_analysis_output_dir, "comparison_analysis")
        
        # Create all directories
        os.makedirs(self.cross_analysis_output_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        os.makedirs(self.execution_time_dir, exist_ok=True)
        os.makedirs(self.utilization_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Regex to extract the 'XX' (Number of Inputs) from 'workload_..._XX-YY.csv' or 'workload_..._XX.csv'
        self.workload_inputs_parser_regex = re.compile(r"_(\d+)(?:-\d+)?\.csv")
        # Regex to extract the 'TTT' (runtime) from 'workload_..._TTTus_...'
        self.workload_runtime_parser_regex = re.compile(r"_(\d+)us_")
    
    def analyze(self):
        """
        Perform complete cross-simulation analysis workflow.
        
        This method loads all result directories, extracts metrics, and generates
        all comparison plots.
        """
        print("\n" + "="*80)
        print("🔬 STARTING CROSS-SIMULATION ANALYSIS")
        print("="*80)
        print(f"📊 Analysis Name: {self.analysis_name}")
        print(f"📁 Number of Result Directories: {len(self.result_directories)}")
        print(f"💾 Output Directory: {self.cross_analysis_output_dir}")
        print("="*80)
        
        # Load all results
        self._load_all_results()
        
        if not self.collected_analysis_data:
            print("❌ No data loaded successfully. Cannot perform analysis.")
            return
        
        # Perform all analyses
        print("\n🔄 Generating comparison plots...")
        self.plot_model_performance_vs_inputs()
        self.plot_execution_time_vs_inputs_by_topology()
        self.plot_execution_time_vs_inputs_by_chiplet_mapping()
        self.plot_avg_model_latency_vs_inputs_by_topology()
        self.plot_execution_time_percent_difference_by_topology()
        self.plot_utilization_vs_inputs()
        self.plot_latency_vs_utilization()
        
        print("\n" + "="*80)
        print("🏁 CROSS-SIMULATION ANALYSIS COMPLETE")
        print("="*80)
        print(f"📁 Results saved to: {self.cross_analysis_output_dir}")
        print("="*80)
    
    def _load_all_results(self):
        """Load metric computers and config info from all result directories."""
        print("\n🔄 Loading results from directories...")
        
        for i, results_dir in enumerate(self.result_directories, 1):
            print(f"\n  [{i}/{len(self.result_directories)}] Loading: {os.path.basename(results_dir)}")
            
            try:
                data_item = self._load_single_result(results_dir)
                if data_item:
                    self.collected_analysis_data.append(data_item)
                    print(f"  ✅ Loaded successfully")
                else:
                    print(f"  ⚠️  Failed to load (see errors above)")
            except Exception as e:
                print(f"  ❌ Error loading result: {e}")
        
        print(f"\n✅ Successfully loaded {len(self.collected_analysis_data)} result(s)")
    
    def _load_single_result(self, results_dir):
        """
        Load a single simulation result.
        
        Args:
            results_dir (str): Path to the results directory (either raw or formatted)
            
        Returns:
            dict: Dictionary containing config_info and metric_computer, or None if failed
        """
        # Determine if this is a raw results directory or formatted results directory
        # Raw results contain .pkl files, formatted results don't
        pkl_files = glob.glob(os.path.join(results_dir, "*.pkl"))
        
        if pkl_files:
            # This is a raw results directory - need to process it
            print(f"    ℹ️  Detected raw results directory, processing...")
            return self._load_from_raw_results(results_dir)
        else:
            # This is a formatted results directory - load metric computer from there
            # (This would require the formatted results to also save the metric computer)
            print(f"    ⚠️  Formatted results directory detected - not yet supported")
            print(f"    💡 Please provide raw results directories for cross-analysis")
            return None
    
    def _load_from_raw_results(self, raw_results_dir):
        """
        Load and process a raw results directory.
        
        Args:
            raw_results_dir (str): Path to raw results directory
            
        Returns:
            dict: Dictionary containing config_info and metric_computer
        """
        from src.post.metrics import MetricComputer
        
        # Find and load the .pkl file
        pkl_files = glob.glob(os.path.join(raw_results_dir, "*.pkl"))
        
        if len(pkl_files) != 1:
            print(f"    ❌ Expected 1 .pkl file, found {len(pkl_files)}")
            return None
        
        pkl_file = pkl_files[0]
        
        # Load the global manager
        try:
            with open(pkl_file, 'rb') as f:
                gm = pickle.load(f)
        except Exception as e:
            print(f"    ❌ Failed to load pickle file: {e}")
            return None
        
        # Migrate legacy attributes
        if hasattr(gm, '_migrate_legacy_attributes'):
            gm._migrate_legacy_attributes()
        
        # Apply warmup filter (using default warmup period from gm)
        warmup_period = getattr(gm, 'warmup_period_us', 0.0)
        gm.warmup_period_us = warmup_period
        gm.filter_retired_networks_by_warmup()
        
        # Create metric computer
        dsent_stats_path = getattr(gm, 'dsent_stats_file', None)
        if not dsent_stats_path or not os.path.exists(dsent_stats_path):
            pkl_path = os.path.join(raw_results_dir, 'dsent_stats.pkl')
            legacy_path = os.path.join(raw_results_dir, 'dsent_stats.jsonl')
            if os.path.exists(pkl_path):
                dsent_stats_path = pkl_path
            elif os.path.exists(legacy_path):
                dsent_stats_path = legacy_path
            else:
                dsent_stats_path = pkl_path
        
        try:
            metric_computer = MetricComputer(
                gm.post_warmup_retired_models,
                gm.global_time_us,
                gm.system.num_chiplets,
                dsent_stats_file_path=dsent_stats_path
            )
            
            # Compute necessary metrics
            metric_computer.compute_avg_system_utilization()
            metric_computer.compute_utilization_over_time(gm.time_step_us)
            metric_computer.compute_model_summary_metrics()
        except Exception as e:
            print(f"    ❌ Failed to compute metrics: {e}")
            return None
        
        # Extract config info from directory name and global manager
        config_info = {
            "workload_name": os.path.basename(gm.workload_manager.wl_file),
            "comm_simulator": gm.communication_simulator,
            "comm_method": gm.communication_method,
            "adj_matrix_file": os.path.basename(gm.adj_matrix_file),
            "chiplet_mapping_file": os.path.basename(gm.chiplet_mapping_file),
            "match_identifier": os.path.basename(raw_results_dir),
            "results_dir": raw_results_dir
        }
        
        return {
            "config_info": config_info,
            "metric_computer": metric_computer
        }


    def _plot_line_chart(self, series_data_map, x_label, y_label, title, filename, 
                         y_log_scale=False, integer_x_axis=False, y_axis_add_plus_sign=False, 
                         figsize=(4.5, 3.2), y_lim=None, x_ticks=None):
        """
        Helper function to generate a line plot for cross-simulation analysis.
        Publication-quality, single-column (research paper) format.

        Args:
            series_data_map (dict): Dict where keys are series labels (str) and 
                                    values are tuples ([x_values], [y_values]).
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            title (str): Title of the plot.
            filename (str): Full path to save the plot.
            y_log_scale (bool): If True, the Y-axis will be on a log scale.
            integer_x_axis (bool): If True, validates and sets the X-axis to use integer ticks.
            y_axis_add_plus_sign (bool): If True, adds a '+' to positive Y-axis tick labels.
            figsize (tuple): Figure size (width, height) in inches.
            y_lim (tuple, optional): A tuple for y-axis limits (min, max). Defaults to None.
            x_ticks (list, optional): A list of explicit x-axis ticks. Defaults to None.
        """
        plt.style.use('seaborn-v0_8-whitegrid') # Using a style for better base aesthetics
        plt.figure(figsize=figsize)
        ax = plt.gca() # Get current axes

        # Set color palette to Set2
        set2_cmap = plt.cm.get_cmap('Set2')
        set2_colors = [set2_cmap(i) for i in range(8)]
        ax.set_prop_cycle(color=set2_colors)

        # Define font sizes for single-column readability
        label_fontsize = 9
        legend_fontsize = 9
        tick_fontsize = 8
        line_linewidth = 2.0
        marker_size = 5
        spine_linewidth = 2.0  # Bold outline
        
        # Define a list of markers
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        
        all_empty = True
        for i, (series_label, (x_values, y_values)) in enumerate(series_data_map.items()):
            if integer_x_axis:
                for x in x_values:
                    if x != int(x):
                        raise ValueError(f"Non-integer x-value found ('{x}') when integer_x_axis is True for plot '{title}'.")

            if x_values and y_values:
                all_empty = False
                sorted_points = sorted(zip(x_values, y_values))
                if not sorted_points:
                    print(f"  ⚠️ No data points for series '{series_label}' in '{title}' after sorting. Skipping this series.")
                    continue
                x_sorted, y_sorted = zip(*sorted_points)
                marker_style = markers[i % len(markers)]
                plt.plot(x_sorted, y_sorted, marker=marker_style, linestyle='-', label=series_label, 
                         linewidth=line_linewidth, markersize=marker_size, 
                         markeredgecolor='black', markeredgewidth=0.8)
            else:
                print(f"  ⚠️ Empty x or y values for series '{series_label}' in '{title}'. Skipping this series.")

        if all_empty:
            print(f"  ⚠️ No data to plot for '{title}'. Skipping plot generation.")
            plt.close() 
            return

        if y_log_scale:
            plt.yscale('log')
        
        if y_lim:
            plt.ylim(y_lim)

        plt.xlabel(x_label, fontsize=label_fontsize, fontweight='bold', labelpad=2)
        plt.ylabel(y_label, fontsize=label_fontsize, fontweight='bold', labelpad=2)
        # No plot title for publication
        # plt.title(title, fontsize=title_fontsize, fontweight='bold', pad=6)
        
        # Only show legend if more than one data series
        if len(series_data_map) > 1:
            legend = plt.legend(fontsize=legend_fontsize, title_fontsize=legend_fontsize, loc='lower center',
                                bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, borderaxespad=0.2)
            if legend.get_title():
                legend.get_title().set_fontweight('bold')
            for text in legend.get_texts():
                text.set_fontweight('bold')
            # Set legend frame color to black
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.0)

        # Customize ticks (make bold)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, length=3, width=0.8)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        elif integer_x_axis:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        if y_axis_add_plus_sign:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}'))

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Thicken plot spines (outlines) - bold and black
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_color('black')
        
        # Ensure grid is present (seaborn-whitegrid adds it, but good to be explicit if needed)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)

        try:
            plt.tight_layout(pad=0.5) # Adjust plot to ensure everything fits
            # Save as both PNG and PDF
            base_filename, _ = os.path.splitext(filename)
            png_filename = base_filename + '.png'
            pdf_filename = base_filename + '.pdf'
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
            print(f"  📊 Saved cross-simulation plot: {os.path.basename(png_filename)} and {os.path.basename(pdf_filename)} to {os.path.dirname(filename)}")
        except Exception as e:
            print(f"  ❌ Error saving cross-simulation plot {os.path.basename(filename)}: {e}")
        plt.close()

    def _filter_data(self, target_runtime_us_str=None, always_pipelined=True):
        """
        Internal helper to filter collected_analysis_data based on runtime and pipelined status.
        """
        if not self.collected_analysis_data:
            print("⚠️ No data provided to CrossSimProcessor for filtering.")
            return []

        filtered_analysis_data = []
        for data_item in self.collected_analysis_data:
            config = data_item["config_info"]
            
            # --- Pipelined Filter ---
            if always_pipelined:
                comm_method_original = config.get("comm_method", "")
                comm_method_lower = comm_method_original.lower()

                # Use exact string matching for correctness
                is_pipelined = comm_method_lower == "pipelined"
                is_non_pipelined = comm_method_lower == "non-pipelined"

                # Validate that comm_method is one of the expected types
                if not is_pipelined and not is_non_pipelined:
                    workload_name = config.get("workload_name", "Unknown Workload")
                    print(f"  ⚠️ Unexpected communication method '{comm_method_original}' found for workload '{workload_name}'. Skipping this data item.")
                    continue
                
                # If we only want pipelined results, and this is not one, skip it.
                if not is_pipelined:
                    continue

            # --- Runtime Filter ---
            if target_runtime_us_str:
                workload_name = config.get("workload_name", "")
                runtime_match = self.workload_runtime_parser_regex.search(workload_name)
                if runtime_match:
                    extracted_runtime_val_str = runtime_match.group(1)
                    normalized_target_runtime_str = target_runtime_us_str.lower().replace("us", "")
                    if extracted_runtime_val_str != normalized_target_runtime_str:
                        continue # Skip items with non-matching runtime
                else:
                    continue # Skip items where runtime cannot be parsed from the name

            # If we reached here, the item passed all filters
            filtered_analysis_data.append(data_item)
            
        return filtered_analysis_data

    def plot_model_performance_vs_inputs(self, target_runtime_us_str=None):
        
        # TODO: Fix in the future
        empty_system_individual_model_results = { 
            }
        
        """
        Generates plots showing average total, compute, and communication time 
        per input for each model type, against the number of inputs from the workload.
        This method ALWAYS considers only pipelined simulation results.

        Args:
            target_runtime_us_str (str, optional): Target runtime to filter by (e.g., "50us", "100us"). 
                                                 Defaults to None (no runtime filtering).
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for model performance vs. inputs (Pipelined Results Only)...")
        
        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for model performance.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for model performance plots after filtering.")

        # --- Data Extraction and Preparation for Model Type Plots (operates on filtered_analysis_data) ---
        raw_plot_points = {"total": {}, "compute": {}, "comm": {}, "comm_absolute": {}}

        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "") # Get it again for parsing inputs number

            workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
            
            if not workload_inputs_match:
                print(f"  ⚠️ Could not parse x-value (Number of Inputs) from workload: {workload_name}. Skipping this data_item.")
                continue
            try:
                x_value_from_workload = int(workload_inputs_match.group(1))
            except ValueError:
                print(f"  ⚠️ Could not convert parsed x-value (Number of Inputs) '{workload_inputs_match.group(1)}' to int for workload: {workload_name}. Skipping this data_item.")
                continue

            instance_summaries = getattr(mc, 'model_summary_metrics', None)
            retired_models_map = getattr(mc, 'retired_mapped_models', None)

            if not instance_summaries or not retired_models_map:
                print(f"  ⚠️ mc.model_summary_metrics or mc.retired_mapped_models not found for workload {workload_name}. Skipping this data_item.")
                continue

            for model_idx, summary_data in instance_summaries.items():
                model_instance = retired_models_map.get(model_idx)
                if not model_instance:
                    # This case should ideally not happen if data is consistent
                    print(f"  ⚠️ Model instance {model_idx} not found in retired_mapped_models for workload {workload_name}. Skipping this model instance metric.")
                    continue
                
                model_type_name = getattr(model_instance, 'model_name', f"UnknownModel_{model_idx}")
                num_inputs_for_instance = getattr(model_instance, 'num_inputs', 0)

                if num_inputs_for_instance <= 0:
                    print(f"  ⚠️ Invalid or zero num_inputs ({num_inputs_for_instance}) for model {model_type_name} (idx {model_idx}) in workload {workload_name}. Skipping this model instance's data.")
                    continue
                
                sum_compute_for_instance = summary_data.get('sum_compute_latency_all_inputs', 0.0)
                sum_activation_comm_for_instance = summary_data.get('sum_activation_comm_latency_all_inputs', 0.0)
                sum_weight_loading_for_instance = summary_data.get('sum_weight_loading_latency_all_inputs', 0.0)
                
                # Combine activation communication and weight loading for total communication time
                sum_comm_for_instance = sum_activation_comm_for_instance + sum_weight_loading_for_instance

                # Calculate per-input latency by averaging weight loading time across all inputs
                avg_compute_this_instance = sum_compute_for_instance / num_inputs_for_instance
                avg_activation_comm_this_instance = sum_activation_comm_for_instance / num_inputs_for_instance
                avg_weight_loading_this_instance = sum_weight_loading_for_instance / num_inputs_for_instance
                avg_total_this_instance = avg_compute_this_instance + avg_activation_comm_this_instance + avg_weight_loading_this_instance
                
                # For communication time, include both activation communication and averaged weight loading
                avg_comm_this_instance = avg_activation_comm_this_instance + avg_weight_loading_this_instance

                # Store total time as an absolute value
                if model_type_name not in raw_plot_points["total"]:
                    raw_plot_points["total"][model_type_name] = []
                raw_plot_points["total"][model_type_name].append((x_value_from_workload, avg_total_this_instance))

                # Calculate and store compute and communication as percentages of total
                if avg_total_this_instance > 0:
                    compute_pct = (avg_compute_this_instance / avg_total_this_instance) * 100
                    comm_pct = (avg_comm_this_instance / avg_total_this_instance) * 100
                else:
                    compute_pct, comm_pct = 0, 0

                if model_type_name not in raw_plot_points["compute"]:
                    raw_plot_points["compute"][model_type_name] = []
                raw_plot_points["compute"][model_type_name].append((x_value_from_workload, compute_pct))

                if model_type_name not in raw_plot_points["comm"]:
                    raw_plot_points["comm"][model_type_name] = []
                raw_plot_points["comm"][model_type_name].append((x_value_from_workload, comm_pct))
                
                # Also store absolute communication latency for the combined plot
                if model_type_name not in raw_plot_points["comm_absolute"]:
                    raw_plot_points["comm_absolute"][model_type_name] = []
                raw_plot_points["comm_absolute"][model_type_name].append((x_value_from_workload, avg_comm_this_instance))

        if not any(raw_plot_points[cat] for cat in ["total", "compute", "comm"]):
            print("⚠️ No data successfully processed for cross-simulation model type line plots after filtering. Skipping plot generation.")
            print("="*80)
            return

        x_axis_label = "Num. of Inferences Per Model Instance"
        plot_configs = {
            "total": {
                "y_label": "Avg. Model Latency (µs)",
                "title": "Cross-Simulation: Avg Total Time vs. Number of Inputs (per Model Type)",
                "filename": "cross_sim_model_avg_total_time.png"
            },
            "compute": {
                "y_label": "Compute Latency (% of Total)",
                "title": "Cross-Simulation: Avg Compute Time vs. Number of Inputs (per Model Type)",
                "filename": "cross_sim_model_avg_compute_time.png"
            },
            "comm": {
                "y_label": "Comm. Latency (% of Total)",
                "title": "Cross-Simulation: Avg Communication Time vs. Number of Inputs (per Model Type)",
                "filename": "cross_sim_model_avg_comm_time.png"
            }
        }

        # Process comm_absolute data FIRST, outside the loop
        final_comm_absolute_series_map = {}
        print(f"  DEBUG: Processing comm_absolute BEFORE category loop")
        print(f"  DEBUG: raw_plot_points keys: {raw_plot_points.keys()}")
        if 'comm_absolute' in raw_plot_points:
            print(f"  DEBUG: comm_absolute has {len(raw_plot_points['comm_absolute'])} models")
            for model_type, points_list in raw_plot_points["comm_absolute"].items():
                if not points_list:
                    continue
                
                points_by_x = {}
                for x, y in points_list:
                    if x not in points_by_x:
                        points_by_x[x] = []
                    points_by_x[x].append(y)
                
                final_x_values = []
                final_y_values = []
                for x_val in sorted(points_by_x.keys()):
                    final_x_values.append(x_val)
                    final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))
                
                if final_x_values:
                    final_comm_absolute_series_map[model_type] = (final_x_values, final_y_values)
                    print(f"    Processed {model_type}: {len(final_x_values)} points")

        for category, config_data in plot_configs.items(): 
            final_series_data_map_for_plot = {}
            if not raw_plot_points[category]:
                print(f"  ⚠️ No data collected for '{category}' time plot. Skipping.")
                continue

            for model_type, points_list in raw_plot_points[category].items():
                if not points_list:
                    print(f"  ⚠️ No points for model type '{model_type}' in '{category}' plot. Skipping this series.")
                    continue
                
                points_by_x = {}
                for x, y in points_list:
                    if x not in points_by_x:
                        points_by_x[x] = []
                    points_by_x[x].append(y)
                
                final_x_values = []
                final_y_values = []
                for x_val in sorted(points_by_x.keys()):
                    final_x_values.append(x_val)
                    final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))

                if final_x_values:
                    final_series_data_map_for_plot[model_type] = (final_x_values, final_y_values)

            if category == 'total' and final_series_data_map_for_plot:
                # --- Create a combined plot for latency change ---
                total_baseline_series_map = {}
                comm_baseline_series_map = {}
                
                # comm_absolute data was already processed before the loop
                print(f"  DEBUG: Using pre-processed comm_absolute with {len(final_comm_absolute_series_map)} models")

                for model_type, (x_vals, y_vals) in final_series_data_map_for_plot.items():
                    if not y_vals or not x_vals:
                        continue
                    
                    print(f"\n📊 Processing {model_type}:")
                    print(f"  X values (num inputs): {x_vals}")
                    print(f"  Y values (total latency): {y_vals}")

                    # --- Right Plot Data (Baseline: Total Latency) ---
                    # Calculate baseline by splitting weight loading time across inputs
                    baseline_data = empty_system_individual_model_results.get(model_type, {}).get("latency", {})
                    baseline_compute = baseline_data.get("compute", 0.0)
                    baseline_activation_comm = baseline_data.get("activation_comm", 0.0)
                    baseline_weight_loading = baseline_data.get("weight_loading", 0.0)
                    
                    if baseline_compute > 0 or baseline_activation_comm > 0 or baseline_weight_loading > 0:
                        # Calculate baseline per-input latency by splitting weight loading across inputs
                        new_y_vals_total = []
                        for x_val in x_vals:
                            # Split weight loading time across the number of inputs
                            baseline_weight_loading_per_input = baseline_weight_loading / x_val if x_val > 0 else 0.0
                            baseline_total_per_input = baseline_compute + baseline_activation_comm + baseline_weight_loading_per_input
                            
                            # Find corresponding y_val for this x_val
                            y_val = y_vals[x_vals.index(x_val)] if x_val in x_vals else 0.0
                            
                            if baseline_total_per_input > 0:
                                percent_change = ((y_val - baseline_total_per_input) / baseline_total_per_input) * 100
                                new_y_vals_total.append(percent_change)
                                if len(new_y_vals_total) <= 3:  # Print first few
                                    print(f"    x={x_val}: actual={y_val:.2f}, baseline={baseline_total_per_input:.2f}, %change={percent_change:+.2f}%")
                            else:
                                new_y_vals_total.append(0.0)
                        
                        total_baseline_series_map[model_type] = (x_vals, new_y_vals_total)
                    else:
                        print(f"  ⚠️ Baseline latency data for {model_type} is not positive. Cannot calculate percentage change.")
                        total_baseline_series_map[model_type] = (x_vals, [0] * len(y_vals))

                    # --- Left Plot Data (Baseline: Communication Latency) ---
                    # Get the actual communication values from the comm_absolute data
                    comm_x_vals, comm_y_vals = final_comm_absolute_series_map.get(model_type, ([], []))
                    
                    print(f"  Communication data for {model_type}:")
                    print(f"    comm_x_vals: {comm_x_vals}")
                    print(f"    comm_y_vals: {comm_y_vals}")
                    
                    if comm_x_vals and comm_y_vals:
                        # Calculate baseline communication by splitting weight loading time across inputs
                        baseline_activation_comm = baseline_data.get("activation_comm", 0.0)
                        baseline_weight_loading = baseline_data.get("weight_loading", 0.0)
                        
                        if baseline_activation_comm > 0 or baseline_weight_loading > 0:
                            # Calculate baseline per-input communication latency by splitting weight loading across inputs
                            new_y_vals_comm = []
                            for i, x_val in enumerate(comm_x_vals):
                                # Split weight loading time across the number of inputs
                                baseline_weight_loading_per_input = baseline_weight_loading / x_val if x_val > 0 else 0.0
                                baseline_comm_per_input = baseline_activation_comm + baseline_weight_loading_per_input
                                
                                # Use the actual communication latency value
                                comm_val = comm_y_vals[i] if i < len(comm_y_vals) else 0.0
                                
                                if baseline_comm_per_input > 0:
                                    percent_change = ((comm_val - baseline_comm_per_input) / baseline_comm_per_input) * 100
                                    new_y_vals_comm.append(percent_change)
                                else:
                                    new_y_vals_comm.append(0.0)
                            
                            comm_baseline_series_map[model_type] = (comm_x_vals, new_y_vals_comm)
                        else:
                            print(f"  ⚠️ Baseline communication latency data for {model_type} is not positive. Cannot calculate percentage change.")
                            comm_baseline_series_map[model_type] = (comm_x_vals, [0] * len(comm_y_vals))
                    else:
                        print(f"  ⚠️ No communication data found for {model_type}. Skipping communication baseline comparison.")

                if total_baseline_series_map or comm_baseline_series_map:
                    custom_x_ticks = [1, 3, 5, 10, 20]
                    
                    self._plot_combined_latency_plots(
                        comm_baseline_series_map,  # Left plot: communication only
                        total_baseline_series_map,  # Right plot: total (compute + comm)
                        x_ticks=custom_x_ticks
                    )
                else:
                    print(f"  ⚠️ No series data to plot for the combined latency plot.")
                
                # --- Original single plot generation is now replaced by the combined plot ---
                continue # Skip the generic plot generation for 'total'

            if final_series_data_map_for_plot:
                all_x_values = [x for x_vals, _ in final_series_data_map_for_plot.values() for x in x_vals]
                if all_x_values:
                    max_x = max(all_x_values)
                    custom_x_ticks = [1] + list(range(5, max_x + 1, 5))
                else:
                    custom_x_ticks = None
                
                y_lim_to_use = None
                if category in ["compute", "comm"]:
                    y_lim_to_use = (0, 101)
                
                add_plus_sign = category == 'total'

                self._plot_line_chart(
                    series_data_map=final_series_data_map_for_plot,
                    x_label=x_axis_label,
                    y_label=config_data["y_label"],
                    title=config_data["title"],
                    filename=os.path.join(self.performance_dir, config_data["filename"]),
                    y_log_scale=False,
                    integer_x_axis=True,
                    figsize=(4.5, 2.38),
                    y_lim=y_lim_to_use,
                    x_ticks=custom_x_ticks,
                    y_axis_add_plus_sign=add_plus_sign
                )
            else:
                print(f"  ⚠️ No series data to plot for '{category}' time after processing all model types.")
                
        print("✅ Cross-simulation model type plots generated if data was available.")
        print("="*80)

    def _plot_combined_latency_plots(self, comm_series_map, total_series_map, x_ticks):
        """
        Generates a side-by-side plot comparing latency changes against communication and total baselines.
        Left plot: Communication only baseline comparison
        Right plot: Total (compute + communication) baseline comparison
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        figsize = (4.9, 2) 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)

        # Common plot elements
        label_fontsize = 9
        legend_fontsize = 9
        tick_fontsize = 8
        line_linewidth = 2.0
        marker_size = 5
        spine_linewidth = 2.0
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        set2_cmap = plt.cm.get_cmap('Set2')
        set2_colors = [set2_cmap(i) for i in range(8)]
        
        plot_data = [
            (ax1, comm_series_map, "Percent Inaccuracy \nComm. Only (%)"),
            (ax2, total_series_map, "Percent Inaccuracy \nCompute + Comm. (%)")
        ]

        for ax, series_map, y_label in plot_data:
            if not series_map:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=label_fontsize)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.set_prop_cycle(color=set2_colors)
            for i, (series_label, (x_values, y_values)) in enumerate(series_map.items()):
                if x_values and y_values:
                    sorted_points = sorted(zip(x_values, y_values))
                    x_sorted, y_sorted = zip(*sorted_points)
                    marker_style = markers[i % len(markers)]
                    ax.plot(x_sorted, y_sorted, marker=marker_style, linestyle='-', label=series_label,
                            linewidth=line_linewidth, markersize=marker_size,
                            markeredgecolor='black', markeredgewidth=0.8)

            ax.set_ylabel(y_label, fontsize=label_fontsize, fontweight='bold', labelpad=2)

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, length=3, width=0.8)
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}'))

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_color('black')
            ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)

        fig.supxlabel("Num. of Inferences Per Model Instance", fontsize=label_fontsize, fontweight='bold')

        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            for label in l:
                if label not in labels:
                    labels.append(label)
                    handles.append(h[l.index(label)])

        if handles and labels:
            legend = fig.legend(handles, labels, fontsize=legend_fontsize, title_fontsize=legend_fontsize,
                                loc='lower center', bbox_to_anchor=(0.5, 0.98), 
                                ncol=len(labels), frameon=True, borderaxespad=0.2)
            if legend.get_title():
                legend.get_title().set_fontweight('bold')
            for text in legend.get_texts():
                text.set_fontweight('bold')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.0)
        
        filename = os.path.join(self.performance_dir, "cross_sim_model_latency_change_combined.png")
        try:
            plt.tight_layout(pad=0.5, rect=(0, 0, 1, 0.95)) # Adjust for figure legend
            base_filename, _ = os.path.splitext(filename)
            png_filename = base_filename + '.png'
            pdf_filename = base_filename + '.pdf'
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
            print(f"  📊 Saved combined cross-simulation plot: {os.path.basename(png_filename)} and {os.path.basename(pdf_filename)}")
        except Exception as e:
            print(f"  ❌ Error saving combined cross-simulation plot {os.path.basename(filename)}: {e}")
        plt.close()
    
    def plot_execution_time_vs_inputs_by_topology(self, target_runtime_us_str=None):
        """
        Generates a plot showing total simulation execution time against the number of inputs,
        with separate lines for different adjacency matrix files (topologies).
        This method ALWAYS considers only pipelined simulation results.
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for Total Execution Time vs. Inputs by Topology (Pipelined Results Only)...")

        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for total execution time.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for total execution time plots after filtering.")

        raw_plot_points = {}

        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "")
            adj_matrix_file = config.get("adj_matrix_file", "unknown_topology")

            workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
            if not workload_inputs_match:
                print(f"  ⚠️ Could not parse x-value (Number of Inputs) from workload: {workload_name}. Skipping this data_item.")
                continue
            try:
                x_value_from_workload = int(workload_inputs_match.group(1))
            except ValueError:
                print(f"  ⚠️ Could not convert parsed x-value '{workload_inputs_match.group(1)}' to int for workload: {workload_name}. Skipping.")
                continue

            total_execution_time = getattr(mc, 'total_simulation_time_us', None)
            if total_execution_time is None:
                print(f"  ⚠️ 'total_simulation_time_us' not found in metric_computer for workload {workload_name}. Skipping.")
                continue

            if adj_matrix_file not in raw_plot_points:
                raw_plot_points[adj_matrix_file] = []
            raw_plot_points[adj_matrix_file].append((x_value_from_workload, total_execution_time))

        if not raw_plot_points:
            print("⚠️ No data successfully processed for total execution time plot. Skipping plot generation.")
            print("="*80)
            return
            
        final_series_data_map_for_plot = {}
        for adj_matrix, points_list in raw_plot_points.items():
            if not points_list:
                continue
            
            points_by_x = {}
            for x, y in points_list:
                if x not in points_by_x:
                    points_by_x[x] = []
                points_by_x[x].append(y)
            
            final_x_values = []
            final_y_values = []
            for x_val in sorted(points_by_x.keys()):
                final_x_values.append(x_val)
                final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))

            if final_x_values:
                series_label = os.path.splitext(adj_matrix)[0]
                final_series_data_map_for_plot[series_label] = (final_x_values, final_y_values)

        if final_series_data_map_for_plot:
            self._plot_line_chart(
                series_data_map=final_series_data_map_for_plot,
                x_label="Num. of Inferences Per Model Instance",
                y_label="Total Execution Time (µs)",
                title="Cross-Sim: Total Execution Time vs. Number of Inputs by Topology",
                filename=os.path.join(self.execution_time_dir, "cross_sim_total_execution_time_by_topology.png"),
                y_log_scale=True,
                integer_x_axis=True
            )
        else:
            print("  ⚠️ No series data to plot for total execution time.")
            
        print("✅ Cross-simulation total execution time plot generated if data was available.")
        print("="*80)

    def plot_execution_time_vs_inputs_by_chiplet_mapping(self, target_runtime_us_str=None):
        """
        Generates a plot showing total simulation execution time against the number of inputs,
        with separate lines for different chiplet mapping files.
        This method ALWAYS considers only pipelined simulation results.
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for Total Execution Time vs. Inputs by Chiplet Mapping (Pipelined Results Only)...")

        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for total execution time.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for total execution time plots after filtering.")

        raw_plot_points = {}

        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "")
            chiplet_mapping_file = config.get("chiplet_mapping_file", "unknown_mapping")

            workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
            if not workload_inputs_match:
                print(f"  ⚠️ Could not parse x-value (Number of Inputs) from workload: {workload_name}. Skipping this data_item.")
                continue
            try:
                x_value_from_workload = int(workload_inputs_match.group(1))
            except ValueError:
                print(f"  ⚠️ Could not convert parsed x-value '{workload_inputs_match.group(1)}' to int for workload: {workload_name}. Skipping.")
                continue

            total_execution_time = getattr(mc, 'total_simulation_time_us', None)
            if total_execution_time is None:
                print(f"  ⚠️ 'total_simulation_time_us' not found in metric_computer for workload {workload_name}. Skipping.")
                continue

            if chiplet_mapping_file not in raw_plot_points:
                raw_plot_points[chiplet_mapping_file] = []
            raw_plot_points[chiplet_mapping_file].append((x_value_from_workload, total_execution_time))

        if not raw_plot_points:
            print("⚠️ No data successfully processed for total execution time plot. Skipping plot generation.")
            print("="*80)
            return
            
        final_series_data_map_for_plot = {}
        # Hardcoded mapping for custom series labels for testing
        custom_labels = {
            "chiplet_mapping_100.yaml": "Homogeneous System",
            "mapping_100_raella-accum_hetero.yaml": "Heterogeneous System"
        }
        for chiplet_mapping, points_list in raw_plot_points.items():
            if not points_list:
                continue
            
            points_by_x = {}
            for x, y in points_list:
                if x not in points_by_x:
                    points_by_x[x] = []
                points_by_x[x].append(y)
            
            final_x_values = []
            final_y_values = []
            for x_val in sorted(points_by_x.keys()):
                final_x_values.append(x_val)
                final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))

            if final_x_values:
                series_label = custom_labels.get(chiplet_mapping, os.path.splitext(chiplet_mapping)[0])
                final_series_data_map_for_plot[series_label] = (final_x_values, final_y_values)

        if final_series_data_map_for_plot:
            self._plot_line_chart(
                series_data_map=final_series_data_map_for_plot,
                x_label="Num. of Inferences Per Model Instance",
                y_label="Total Execution Time (µs)",
                title="Cross-Sim: Total Execution Time vs. Number of Inputs by Chiplet Mapping",
                filename=os.path.join(self.execution_time_dir, "cross_sim_total_execution_time_by_chiplet_mapping.png"),
                y_log_scale=False,
                integer_x_axis=True,
                figsize=(4.5, 2.38)
            )
        else:
            print("  ⚠️ No series data to plot for total execution time.")
            
        print("✅ Cross-simulation total execution time by chiplet mapping plot generated if data was available.")
        print("="*80)

    def plot_avg_model_latency_vs_inputs_by_topology(self, target_runtime_us_str=None):
        """
        Generates a plot showing average model latency against the number of inputs,
        with separate lines for different adjacency matrix files (topologies).
        This method ALWAYS considers only pipelined simulation results.
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for Avg Model Latency vs. Inputs by Topology (Pipelined Results Only)...")

        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for avg model latency.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for avg model latency plots after filtering.")

        raw_plot_points = {}

        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "")
            adj_matrix_file = config.get("adj_matrix_file", "unknown_topology")

            workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
            if not workload_inputs_match:
                print(f"  ⚠️ Could not parse x-value (Number of Inputs) from workload: {workload_name}. Skipping this data_item.")
                continue
            try:
                x_value_from_workload = int(workload_inputs_match.group(1))
            except ValueError:
                print(f"  ⚠️ Could not convert parsed x-value '{workload_inputs_match.group(1)}' to int for workload: {workload_name}. Skipping.")
                continue

            instance_summaries = getattr(mc, 'model_summary_metrics', None)
            retired_models_map = getattr(mc, 'retired_mapped_models', None)

            if not instance_summaries or not retired_models_map:
                print(f"  ⚠️ mc.model_summary_metrics or mc.retired_mapped_models not found for workload {workload_name}. Skipping this data_item.")
                continue

            latencies_for_this_run = []
            for model_idx, summary_data in instance_summaries.items():
                model_instance = retired_models_map.get(model_idx)
                if not model_instance:
                    continue
                
                num_inputs = getattr(model_instance, 'num_inputs', 0)
                if num_inputs <= 0:
                    continue
                
                sum_compute = summary_data.get('sum_compute_latency_all_inputs', 0.0)
                sum_activation_comm = summary_data.get('sum_activation_comm_latency_all_inputs', 0.0)
                sum_weight_loading = summary_data.get('sum_weight_loading_latency_all_inputs', 0.0)
                sum_comm = sum_activation_comm + sum_weight_loading
                avg_total = (sum_compute + sum_comm) / num_inputs
                latencies_for_this_run.append(avg_total)

            if latencies_for_this_run:
                avg_latency_for_run = sum(latencies_for_this_run) / len(latencies_for_this_run)
                if adj_matrix_file not in raw_plot_points:
                    raw_plot_points[adj_matrix_file] = []
                raw_plot_points[adj_matrix_file].append((x_value_from_workload, avg_latency_for_run))

        if not raw_plot_points:
            print("⚠️ No data successfully processed for avg model latency plot. Skipping plot generation.")
            print("="*80)
            return
            
        final_series_data_map_for_plot = {}
        for adj_matrix, points_list in raw_plot_points.items():
            if not points_list:
                continue
            
            points_by_x = {}
            for x, y in points_list:
                if x not in points_by_x:
                    points_by_x[x] = []
                points_by_x[x].append(y)
            
            final_x_values = []
            final_y_values = []
            for x_val in sorted(points_by_x.keys()):
                final_x_values.append(x_val)
                final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))

            if final_x_values:
                series_label = os.path.splitext(adj_matrix)[0]
                final_series_data_map_for_plot[series_label] = (final_x_values, final_y_values)

        if final_series_data_map_for_plot:
            self._plot_line_chart(
                series_data_map=final_series_data_map_for_plot,
                x_label="Num. of Inferences Per Model Instance",
                y_label="Avg. Model Latency (µs)",
                title="Cross-Sim: Avg Model Latency vs. Number of Inputs by Topology",
                filename=os.path.join(self.performance_dir, "cross_sim_avg_model_latency_by_topology.png"),
                y_log_scale=False,
                integer_x_axis=True
            )
        else:
            print("  ⚠️ No series data to plot for average model latency.")
            
        print("✅ Cross-simulation average model latency plot generated if data was available.")
        print("="*80)

    def plot_execution_time_percent_difference_by_topology(self, target_runtime_us_str=None):
        """
        Generates a plot showing the percent difference in total execution time 
        between two topologies against the number of inputs.
        This method is only valid for exactly two topologies and uses pipelined results.
        """
        print("\n" + "="*80)
        print("🔬 Performing analysis for Execution Time % Difference vs. Inputs by Topology (Pipelined)...")
        
        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for percent difference plot after filtering.")

        raw_plot_points = {}
        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "")
            adj_matrix_file = config.get("adj_matrix_file", "unknown_topology")

            workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
            if not workload_inputs_match:
                continue
            try:
                x_value_from_workload = int(workload_inputs_match.group(1))
            except ValueError:
                continue

            total_execution_time = getattr(mc, 'total_simulation_time_us', None)
            if total_execution_time is None:
                continue

            if adj_matrix_file not in raw_plot_points:
                raw_plot_points[adj_matrix_file] = []
            raw_plot_points[adj_matrix_file].append((x_value_from_workload, total_execution_time))

        if len(raw_plot_points) != 2:
            print(f"⚠️  Skipping percent difference plot: this plot requires exactly 2 topologies, but {len(raw_plot_points)} were found.")
            print("="*80)
            return

        processed_points = {}
        for adj_matrix, points_list in raw_plot_points.items():
            points_by_x = {}
            for x, y in points_list:
                if x not in points_by_x:
                    points_by_x[x] = []
                points_by_x[x].append(y)
            
            avg_points = {}
            for x_val in sorted(points_by_x.keys()):
                avg_points[x_val] = sum(points_by_x[x_val]) / len(points_by_x[x_val])
            processed_points[adj_matrix] = avg_points

        topo_names = sorted(list(processed_points.keys()))

        # If both 'mesh' and 'floret' topologies are present, ensure 'mesh' is the baseline.
        is_mesh_present = any('mesh' in name.lower() for name in topo_names)
        is_floret_present = any('floret' in name.lower() for name in topo_names)

        if is_mesh_present and is_floret_present:
            if 'floret' in topo_names[0].lower():
                topo_names.reverse()  # This will make mesh the baseline

        baseline_topo_name = topo_names[0]
        comparison_topo_name = topo_names[1]

        baseline_points = processed_points[baseline_topo_name]
        comparison_points = processed_points[comparison_topo_name]
        
        percent_diff_x = []
        percent_diff_y = []

        common_x_values = sorted(list(set(baseline_points.keys()) & set(comparison_points.keys())))

        if not common_x_values:
            print("⚠️ No common x-values found between the two topologies. Skipping percent difference plot.")
            print("="*80)
            return

        for x in common_x_values:
            baseline_y = baseline_points[x]
            comparison_y = comparison_points[x]
            
            if baseline_y == 0:
                percent_diff = 0
            else:
                percent_diff = ((comparison_y - baseline_y) / baseline_y) * 100
            
            percent_diff_x.append(x)
            percent_diff_y.append(percent_diff)

        baseline_label = os.path.splitext(baseline_topo_name)[0].replace('adj_matrix_', '')
        comparison_label = os.path.splitext(comparison_topo_name)[0].replace('adj_matrix_', '')
        series_label = f"% Change ({comparison_label} vs. {baseline_label})"
        
        series_data_map = {
            series_label: (percent_diff_x, percent_diff_y)
        }

        self._plot_line_chart(
            series_data_map=series_data_map,
            x_label="Num. of Inferences Per Model Instance",
            y_label="Execution Time\nPercent Change (%)",
            title="Cross-Sim: Percent Change in Execution Time by Topology",
            filename=os.path.join(self.comparison_dir, "cross_sim_exec_time_percent_change.png"),
            y_log_scale=False,
            integer_x_axis=True,
            figsize=(4.5, 2.38)
        )
        
        print("✅ Cross-simulation execution time percent change plot generated if data was available.")
        print("="*80)

    def plot_utilization_vs_inputs(self, target_runtime_us_str=None):
        """
        Generates plots showing average system utilization (based on compute, communication,
        and combined time) against the number of inputs from the workload.
        This method ALWAYS considers only pipelined simulation results.

        Args:
            target_runtime_us_str (str, optional): Target runtime to filter by (e.g., "50us", "100us").
                                                 Defaults to None (no runtime filtering).
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for Utilization vs. Inputs (Pipelined Results Only)...")

        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for utilization.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for utilization plots after filtering.")

        utilization_types = {
            "compute": {
                "attr_name": "mean_compute_time_chiplet_utilization_pct",
                "y_label": "Avg. Compute Time Based System Utilization (%)",
                "title_suffix": "Compute Time Utilization",
                "filename_suffix": "compute_time_utilization"
            },
            "communication": {
                "attr_name": "mean_communication_time_chiplet_utilization_pct",
                "y_label": "Avg. Communication Time Based System Utilization (%)",
                "title_suffix": "Communication Time Utilization",
                "filename_suffix": "communication_time_utilization"
            },
            "combined": {
                "attr_name": "mean_combined_time_chiplet_utilization_pct",
                "y_label": "Avg. System \nUtilization (%)",
                "title_suffix": "Combined Time Utilization",
                "filename_suffix": "combined_time_utilization"
            }
        }

        for util_type, type_config in utilization_types.items():
            print(f"Processing {util_type} utilization...")
            utilization_points_by_x = {}
            metric_attr_name = type_config["attr_name"]

            for data_item in filtered_analysis_data:
                config = data_item["config_info"]
                mc = data_item["metric_computer"]
                workload_name = config.get("workload_name", "")

                workload_inputs_match = self.workload_inputs_parser_regex.search(workload_name)
                if not workload_inputs_match:
                    print(f"  ⚠️ Could not parse x-value (Number of Inputs) from workload: {workload_name}. Skipping this data_item for {util_type} utilization.")
                    continue
                try:
                    x_value_from_workload = int(workload_inputs_match.group(1))
                except ValueError:
                    print(f"  ⚠️ Could not convert parsed x-value (Number of Inputs) '{workload_inputs_match.group(1)}' to int for workload: {workload_name}. Skipping for {util_type} utilization.")
                    continue

                utilization_value = getattr(mc, metric_attr_name, None)

                if utilization_value is None:
                    print(f"  ⚠️ Invalid or missing {metric_attr_name} for workload {workload_name} (Value: {utilization_value}). Skipping this data point for {util_type} utilization.")
                    continue
                
                if x_value_from_workload not in utilization_points_by_x:
                    utilization_points_by_x[x_value_from_workload] = []
                utilization_points_by_x[x_value_from_workload].append(utilization_value)

            if not utilization_points_by_x:
                print(f"⚠️ No {util_type} utilization data points collected after processing filtered data. Skipping plot for this type.")
                continue

            final_x_values = []
            final_y_values = []
            for x_val in sorted(utilization_points_by_x.keys()):
                avg_util_for_x = sum(utilization_points_by_x[x_val]) / len(utilization_points_by_x[x_val])
                final_x_values.append(x_val)
                final_y_values.append(avg_util_for_x)
            
            if not final_x_values:
                print(f"⚠️ No data to plot for {util_type} utilization after averaging. Skipping plot for this type.")
                continue

            series_data_map = {
                f"Avg {type_config['title_suffix']} (Pipelined)": (final_x_values, final_y_values)
            }

            self._plot_line_chart(
                series_data_map=series_data_map,
                x_label="Num. of Inferences Per Model Instance",
                y_label=type_config["y_label"],
                title=f"Cross-Sim: Avg {type_config['title_suffix']} vs. Number of Inputs (Pipelined)",
                filename=os.path.join(self.utilization_dir, f"cross_sim_avg_{type_config['filename_suffix']}.png"),
                y_log_scale=False,
                integer_x_axis=True,
                figsize=(4.5, 2.38)
            )
        
        print(f"✅ Cross-simulation {util_type} utilization plot generated if data was available.")
        print("="*80) 

    def plot_latency_vs_utilization(self, target_runtime_us_str=None):
        """
        Generates a plot showing average model latency against average system utilization.
        This method ALWAYS considers only pipelined simulation results.

        Args:
            target_runtime_us_str (str, optional): Target runtime to filter by (e.g., "50us", "100us").
                                                 Defaults to None (no runtime filtering).
        """
        print("\n" + "="*80)
        print("🔬 Performing cross-simulation analysis for Model Latency vs. System Utilization (Pipelined Results Only)...")

        filters_applied_list = ["only_pipelined=True"]
        if target_runtime_us_str:
            filters_applied_list.append(f"target_runtime='{target_runtime_us_str}'")
        filters_applied_str = ", ".join(filters_applied_list)
        print(f"ℹ️ Applying filters: {filters_applied_str}")
        print(f"ℹ️ Cross-simulation analysis results will be saved to: {self.cross_analysis_output_dir}")
        print(f"ℹ️ Organized into subdirectories: performance_analysis/, execution_time_analysis/, utilization_analysis/, comparison_analysis/")

        filtered_analysis_data = self._filter_data(target_runtime_us_str=target_runtime_us_str, always_pipelined=True)

        if not filtered_analysis_data:
            print("⚠️ No data remains after filtering. Skipping plot generation for latency vs. utilization.")
            print("="*80)
            return
        print(f"ℹ️ Proceeding with {len(filtered_analysis_data)} data items for latency vs. utilization plot after filtering.")

        raw_plot_points = {} # {model_type_name: [(utilization, latency), ...]}

        for data_item in filtered_analysis_data:
            config = data_item["config_info"]
            mc = data_item["metric_computer"]
            workload_name = config.get("workload_name", "")
            comm_method = config.get("comm_method", "")

            # Safeguard: Explicitly ensure we are only processing pipelined results.
            if "pipelined" not in comm_method.lower():
                continue

            utilization_value = getattr(mc, "mean_combined_time_chiplet_utilization_pct", None)
            if utilization_value is None:
                print(f"  ⚠️ Invalid or missing mean_combined_time_chiplet_utilization_pct for workload {workload_name}. Skipping this data item.")
                continue

            instance_summaries = getattr(mc, 'model_summary_metrics', None)
            retired_models_map = getattr(mc, 'retired_mapped_models', None)

            if not instance_summaries or not retired_models_map:
                print(f"  ⚠️ mc.model_summary_metrics or mc.retired_mapped_models not found for workload {workload_name}. Skipping this data_item.")
                continue

            for model_idx, summary_data in instance_summaries.items():
                model_instance = retired_models_map.get(model_idx)
                if not model_instance:
                    print(f"  ⚠️ Model instance {model_idx} not found in retired_mapped_models for workload {workload_name}. Skipping this model instance metric.")
                    continue
                
                model_type_name = getattr(model_instance, 'model_name', f"UnknownModel_{model_idx}")
                num_inputs_for_instance = getattr(model_instance, 'num_inputs', 0)

                if num_inputs_for_instance <= 0:
                    print(f"  ⚠️ Invalid or zero num_inputs ({num_inputs_for_instance}) for model {model_type_name} (idx {model_idx}) in workload {workload_name}. Skipping this model instance's data.")
                    continue
                
                sum_compute_for_instance = summary_data.get('sum_compute_latency_all_inputs', 0.0)
                sum_activation_comm_for_instance = summary_data.get('sum_activation_comm_latency_all_inputs', 0.0)
                sum_weight_loading_for_instance = summary_data.get('sum_weight_loading_latency_all_inputs', 0.0)
                sum_comm_for_instance = sum_activation_comm_for_instance + sum_weight_loading_for_instance

                avg_compute_this_instance = sum_compute_for_instance / num_inputs_for_instance
                avg_activation_comm_this_instance = sum_activation_comm_for_instance / num_inputs_for_instance
                avg_weight_loading_this_instance = sum_weight_loading_for_instance / num_inputs_for_instance
                avg_comm_this_instance = avg_activation_comm_this_instance + avg_weight_loading_this_instance
                avg_total_latency = avg_compute_this_instance + avg_comm_this_instance

                if model_type_name not in raw_plot_points:
                    raw_plot_points[model_type_name] = []
                raw_plot_points[model_type_name].append((utilization_value, avg_total_latency))

        if not raw_plot_points:
            print("⚠️ No data successfully processed for latency vs. utilization plot. Skipping plot generation.")
            print("="*80)
            return

        final_series_data_map_for_plot = {}
        for model_type, points_list in raw_plot_points.items():
            if not points_list:
                print(f"  ⚠️ No points for model type '{model_type}' in latency vs. utilization plot. Skipping this series.")
                continue
            
            points_by_x = {}
            for x, y in points_list:
                if x not in points_by_x:
                    points_by_x[x] = []
                points_by_x[x].append(y)
            
            final_x_values = []
            final_y_values = []
            for x_val in sorted(points_by_x.keys()):
                final_x_values.append(x_val)
                final_y_values.append(sum(points_by_x[x_val]) / len(points_by_x[x_val]))

            if final_x_values:
                final_series_data_map_for_plot[model_type] = (final_x_values, final_y_values)

        if final_series_data_map_for_plot:
            self._plot_line_chart(
                series_data_map=final_series_data_map_for_plot,
                x_label="Avg. System Utilization (%)",
                y_label="Avg. Model Latency (µs)",
                title="Cross-Sim: Avg Model Latency vs. System Utilization (Pipelined)",
                filename=os.path.join(self.utilization_dir, "cross_sim_latency_vs_utilization.png"),
                y_log_scale=False,
                figsize=(4.5, 2.38)
            )
        else:
            print(f"  ⚠️ No series data to plot for latency vs. utilization after processing all model types.")
            
        print("✅ Cross-simulation latency vs. utilization plot generated if data was available.")
        print("="*80) 