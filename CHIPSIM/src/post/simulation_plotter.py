import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

class SimulationPlotter:
    """
    Handles the creation of plots to visualize simulation results.
    """
    def __init__(self, results_folder, metric_computer):
        """
        Initializes the SimulationPlotter.

        Args:
            results_folder (str): The path to the main results directory.
            metric_computer (MetricComputer): The computed metrics object.
        """
        self.results_folder = results_folder
        self.metric_computer = metric_computer # Store the metric computer instance
        self.plots_folder = os.path.join(self.results_folder, "plots")
        os.makedirs(self.plots_folder, exist_ok=True)
        print(f"📊 Plots will be saved to: {self.plots_folder}")

    # --- Plotting methods ---

    def plot_utilization_over_time(self, rolling_avg_window=1000):
        """
        Creates plots for system utilization over time based on data 
        retrieved from the stored MetricComputer instance.
        This will generate separate plots for compute, communication, and combined utilization.
        
        Args:
            rolling_avg_window (int, optional): Window size for calculating rolling average.
        """
        print("\n📊 Generating system utilization plots...")

        # Retrieve data from metric_computer
        if not self.metric_computer:
             print("  ⚠️ Error: MetricComputer instance not found.")
             return None
             
        time_points = getattr(self.metric_computer, 'utilization_time_points', None)
        if time_points is None:
            print("  ⚠️ Utilization data not available in MetricComputer. Skipping utilization plots.")
            return None
            
        # Check if data is empty (e.g., simulation didn't run properly)
        if len(time_points) == 0:
            print("  ⚠️ Utilization data is empty. Skipping utilization plots.")
            return None
        
        # Create a subdirectory for utilization plots
        utilization_plots_dir = os.path.join(self.plots_folder, "utilization_plots")
        os.makedirs(utilization_plots_dir, exist_ok=True)
        
        utilization_types = ["compute", "communication", "combined"]

        for util_type in utilization_types:
            system_util_attr = f'system_{util_type}_utilization_over_time_pct'
            chiplet_activity_attr = f'chiplet_{util_type}_activity_over_time'

            system_utilization_values = getattr(self.metric_computer, system_util_attr, None)
            chiplet_activity_values = getattr(self.metric_computer, chiplet_activity_attr, None)

            if system_utilization_values is None or not chiplet_activity_values:
                print(f"  ⚠️ {util_type.capitalize()} utilization data not found in MetricComputer. Skipping these plots.")
                continue

            if len(system_utilization_values) == 0 or not chiplet_activity_values:
                print(f"  ⚠️ {util_type.capitalize()} utilization data is empty. Skipping these plots.")
                continue

            # Plot 1: System utilization over time for the current type
            plt.figure(figsize=(8, 6)) # Adjusted figsize
            ax = plt.gca() # Get current axes

            plt.plot(time_points, system_utilization_values, label=f'Instantaneous {util_type.capitalize()} Utilization')
            
            if rolling_avg_window and rolling_avg_window > 0 and len(system_utilization_values) > rolling_avg_window:
                rolling_avg = np.convolve(system_utilization_values, np.ones(rolling_avg_window)/rolling_avg_window, mode='valid')
                rolling_time_points = time_points[rolling_avg_window-1:]
                plt.plot(rolling_time_points, rolling_avg, 'r-', linewidth=2, label=f'Rolling Average (window={rolling_avg_window})')
                print(f"    📊 Added rolling average for {util_type} utilization with window size {rolling_avg_window}")

            plt.title(f'System Average {util_type.capitalize()} Utilization Over Time', fontweight='bold', fontsize='x-large')
            plt.xlabel('Time (μs)', fontweight='bold', fontsize='large')
            plt.ylabel('Average Utilization (%)', fontweight='bold', fontsize='large')
            plt.grid(True)
            
            # Legend styling
            legend = ax.legend(prop={'weight':'bold', 'size':12}, frameon=True, edgecolor='black', fancybox=False)
            if legend:
                legend.get_frame().set_linewidth(1.5)

            # Spines styling
            for spine_pos in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine_pos].set_linewidth(1.5)

            # Tick labels styling
            ax.tick_params(axis='both', which='major') # Ensure ticks are on
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(11)
            
            plot_filename = f'system_avg_{util_type}_utilization.png'
            plt.savefig(os.path.join(utilization_plots_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  📊 Saved plot: {plot_filename}")

            # Plot 2: Per-chiplet average utilization bar chart for the current type
            chiplet_ids_for_type = sorted(chiplet_activity_values.keys())
            if not chiplet_ids_for_type:
                print(f"  ⚠️ No chiplet activity data for {util_type} utilization. Skipping bar chart.")
                continue

            chiplet_avg_utilization_for_type = {}
            for chiplet_id, activity_list in chiplet_activity_values.items():
                if activity_list: # Ensure list is not empty to avoid division by zero
                    chiplet_avg_utilization_for_type[chiplet_id] = sum(activity_list) / len(activity_list) * 100.0
                else:
                    chiplet_avg_utilization_for_type[chiplet_id] = 0.0 # Or handle as appropriate
            
            sorted_chiplets_for_type = sorted(chiplet_avg_utilization_for_type.keys())
            utilization_values_for_type = [chiplet_avg_utilization_for_type[c] for c in sorted_chiplets_for_type]
            
            plt.figure(figsize=(8, 6)) # Adjusted figsize
            ax_bar = plt.gca() # Get current axes for bar chart

            plt.bar(range(len(sorted_chiplets_for_type)), utilization_values_for_type)
            plt.xticks(range(len(sorted_chiplets_for_type)), [str(c) for c in sorted_chiplets_for_type]) # Set ticks before styling
            
            plt.title(f'Average {util_type.capitalize()} Utilization by Chiplet', fontweight='bold', fontsize='x-large')
            plt.xlabel('Chiplet ID', fontweight='bold', fontsize='large')
            plt.ylabel('Average Utilization (%)', fontweight='bold', fontsize='large')
            plt.grid(True, axis='y')

            # Spines styling for bar chart
            for spine_pos in ['top', 'right', 'bottom', 'left']:
                ax_bar.spines[spine_pos].set_linewidth(1.5)

            # Tick labels styling for bar chart
            ax_bar.tick_params(axis='both', which='major') # Ensure ticks are on
            for label in ax_bar.get_xticklabels(): # Style x-ticks (already set by plt.xticks)
                label.set_fontweight('bold')
                label.set_fontsize(11)
                # Potentially add rotation if needed for many chiplets, e.g., label.set_rotation(45), label.set_ha('right')
            for label in ax_bar.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(11)

            bar_chart_filename = f'chiplet_avg_{util_type}_utilization.png'
            plt.savefig(os.path.join(utilization_plots_dir, bar_chart_filename), dpi=300)
            plt.close()
            print(f"  📊 Saved plot: {bar_chart_filename}")
         
        print(f"📊 System utilization plots saved to: {utilization_plots_dir}")
        
        return utilization_plots_dir

    def plot_power_over_time(self, rolling_avg_window=1000):
        """
        Creates plots for system and chiplet power over time.
        Generates plots for total system power, per-chiplet power, and average power per chiplet,
        for each power type: total, compute, and communication.
        """
        print("\n📊 Generating power profile plots...")

        if not self.metric_computer:
            print("  ⚠️ Error: MetricComputer instance not found.")
            return None

        time_points = getattr(self.metric_computer, 'power_time_points', None)
        if time_points is None or len(time_points) == 0:
            print("  ⚠️ Power data not available or empty in MetricComputer. Skipping power plots.")
            return None
        
        plt.style.use('seaborn-v0_8-whitegrid')

        power_plots_dir = os.path.join(self.plots_folder, "power_plots")
        os.makedirs(power_plots_dir, exist_ok=True)
        # New subdirectories for each plot type
        system_power_dir = os.path.join(power_plots_dir, "system_power")
        chiplet_power_over_time_dir = os.path.join(power_plots_dir, "chiplet_power_over_time")
        chiplet_selected_power_over_time_dir = os.path.join(power_plots_dir, "chiplet_selected_power_over_time")
        chiplet_average_power_dir = os.path.join(power_plots_dir, "chiplet_average_power")
        combined_power_dir = os.path.join(power_plots_dir, "combined_power")
        os.makedirs(system_power_dir, exist_ok=True)
        os.makedirs(chiplet_power_over_time_dir, exist_ok=True)
        os.makedirs(chiplet_selected_power_over_time_dir, exist_ok=True)
        os.makedirs(chiplet_average_power_dir, exist_ok=True)
        os.makedirs(combined_power_dir, exist_ok=True)
        
        power_types = ["total", "compute", "communication"]

        for power_type in power_types:
            # --- Get data for the current power type ---
            system_power_attr = f'system_{power_type}_power_over_time'
            chiplet_power_attr = f'chiplet_{power_type}_power_over_time'

            system_power = getattr(self.metric_computer, system_power_attr, np.array([]))
            chiplet_power = getattr(self.metric_computer, chiplet_power_attr, {})
            
            # Check if data exists for this power type
            system_power_available = isinstance(system_power, np.ndarray) and system_power.size > 0
            chiplet_power_available = isinstance(chiplet_power, dict) and chiplet_power

            if not system_power_available and not chiplet_power_available:
                print(f"  ⚠️ No {power_type} power data available. Skipping these plots.")
                continue

            print(f"\n  Generating {power_type.capitalize()} power plots...")
            title_power_type = power_type.capitalize()
            
            # --- Plot 1: System Power Over Time ---
            if system_power_available:
                plt.figure(figsize=(4.5, 2.38))
                ax = plt.gca() # Get current axes
                
                plt.plot(time_points, system_power, label=f'System {title_power_type} Power', linewidth=2.0)
                if rolling_avg_window and rolling_avg_window > 0 and len(system_power) > rolling_avg_window:
                    rolling_avg = np.convolve(system_power, np.ones(rolling_avg_window)/rolling_avg_window, mode='valid')
                    rolling_time_points = time_points[rolling_avg_window-1:]
                    plt.plot(rolling_time_points, rolling_avg, 'r-', linewidth=2.0, label=f'Rolling Average')

                plt.xlabel('Time (μs)', fontweight='bold', fontsize=9, labelpad=2)
                plt.ylabel('Power (W)', fontweight='bold', fontsize=9, labelpad=2)
                plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
                
                legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, frameon=True, borderaxespad=0.2)
                if legend:
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_linewidth(1.0)
                    for text in legend.get_texts():
                        text.set_fontweight('bold')

                for spine in ax.spines.values():
                    spine.set_linewidth(2.0)
                    spine.set_color('black')

                ax.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.8)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

                plt.tight_layout(pad=0.5)
                plot_filename_base = f'system_{power_type}_power'
                png_path = os.path.join(system_power_dir, f"{plot_filename_base}.png")
                pdf_path = os.path.join(system_power_dir, f"{plot_filename_base}.pdf")
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    📊 Saved plots: {os.path.join('system_power', plot_filename_base)}.png/pdf")
            else:
                print(f"    ⚠️ System {power_type} power data not available. Skipping system power plot.")

            # --- Plot 2: Per-Chiplet Power Over Time ---
            if chiplet_power_available:
                plt.figure(figsize=(12, 7))
                ax = plt.gca() # Get current axes

                for chiplet_id, power_values in sorted(chiplet_power.items()):
                    if power_values is not None and len(power_values) == len(time_points):
                        plt.plot(time_points, power_values, label=f'Chiplet {chiplet_id}')

                plt.title(f'{title_power_type} Power Over Time per Chiplet', fontweight='bold', fontsize='x-large')
                plt.xlabel('Time (μs)', fontweight='bold', fontsize='large')
                plt.ylabel('Power (W)', fontweight='bold', fontsize='large')
                plt.grid(True)
                
                legend = ax.legend(ncol=min(len(chiplet_power), 5), prop={'weight':'bold', 'size':10}, frameon=True, edgecolor='black', fancybox=False,
                                   loc='center left', bbox_to_anchor=(1, 0.5))
                if legend:
                    legend.get_frame().set_linewidth(1.5)
                
                for spine_pos in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine_pos].set_linewidth(1.5)
                
                ax.tick_params(axis='both', which='major')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                    label.set_fontsize(11)

                plt.tight_layout()
                plot_filename_base = f'chiplet_{power_type}_power_over_time'
                png_path = os.path.join(chiplet_power_over_time_dir, f"{plot_filename_base}.png")
                pdf_path = os.path.join(chiplet_power_over_time_dir, f"{plot_filename_base}.pdf")
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    📊 Saved plots: {os.path.join('chiplet_power_over_time', plot_filename_base)}.png/pdf")
            else:
                print(f"    ⚠️ Chiplet-level {power_type} power data not available. Skipping per-chiplet power plot.")

            # --- Plot 3: Power Over Time for Selected Chiplets ---
            if chiplet_power_available:
                # Hard-coded list of chiplets to plot for detailed analysis
                chiplets_to_plot = [1, 10, 51] 

                # Filter to only include chiplets that exist in the data
                plottable_chiplet_power = {cid: chiplet_power[cid] for cid in chiplets_to_plot if cid in chiplet_power}
                
                if plottable_chiplet_power:
                    plt.figure(figsize=(4.5, 2.38))
                    ax = plt.gca()

                    for chiplet_id, power_values in sorted(plottable_chiplet_power.items()):
                        if power_values is not None and len(power_values) == len(time_points):
                            plt.plot(time_points, power_values, label=f'Chiplet {chiplet_id}', linewidth=2.0)

                    plt.xlabel('Time (μs)', fontweight='bold', fontsize=9, labelpad=2)
                    plt.ylabel('Power (W)', fontweight='bold', fontsize=9, labelpad=2)
                    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
                    
                    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(len(plottable_chiplet_power), 3), fontsize=9,
                                       frameon=True, borderaxespad=0.2)
                    if legend:
                        legend.get_frame().set_edgecolor('black')
                        legend.get_frame().set_linewidth(1.0)
                        for text in legend.get_texts():
                            text.set_fontweight('bold')
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(2.0)
                        spine.set_color('black')
                    
                    ax.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.8)
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontweight('bold')

                    plt.tight_layout(pad=0.5)
                    plot_filename_base = f'selected_chiplet_{power_type}_power_over_time'
                    png_path = os.path.join(chiplet_selected_power_over_time_dir, f"{plot_filename_base}.png")
                    pdf_path = os.path.join(chiplet_selected_power_over_time_dir, f"{plot_filename_base}.pdf")
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"    📊 Saved plots: {os.path.join('chiplet_selected_power_over_time', plot_filename_base)}.png/pdf")
                else:
                    print(f"    ⚠️ None of the selected chiplets for plotting ({chiplets_to_plot}) were found for {power_type} power.")

            # --- Plot 4: Average Power per Chiplet (Bar Chart) ---
            if chiplet_power_available:
                avg_chiplet_power = {cid: np.mean(pwr) for cid, pwr in chiplet_power.items() if pwr is not None and pwr.size > 0}
                if avg_chiplet_power:
                    sorted_chiplets = sorted(avg_chiplet_power.keys())
                    avg_power_values = [avg_chiplet_power[c] for c in sorted_chiplets]
                    
                    plt.figure(figsize=(10, 6))
                    ax_bar = plt.gca()

                    plt.bar([str(c) for c in sorted_chiplets], avg_power_values)
                    plt.xticks(range(len(sorted_chiplets)), [str(c) for c in sorted_chiplets])

                    plt.title(f'Average {title_power_type} Power per Chiplet', fontweight='bold', fontsize='x-large')
                    plt.xlabel('Chiplet ID', fontweight='bold', fontsize='large')
                    plt.ylabel('Average Power (W)', fontweight='bold', fontsize='large')
                    plt.grid(True, axis='y')
                    
                    for spine_pos in ['top', 'right', 'bottom', 'left']:
                        ax_bar.spines[spine_pos].set_linewidth(1.5)

                    ax_bar.tick_params(axis='both', which='major')
                    for label in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
                        label.set_fontweight('bold')
                        label.set_fontsize(11)

                    plot_filename_base = f'chiplet_average_{power_type}_power'
                    png_path = os.path.join(chiplet_average_power_dir, f"{plot_filename_base}.png")
                    pdf_path = os.path.join(chiplet_average_power_dir, f"{plot_filename_base}.pdf")
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"    📊 Saved plots: {os.path.join('chiplet_average_power', plot_filename_base)}.png/pdf")
                else:
                    print(f"    ⚠️ No valid data to calculate average {power_type} power per chiplet.")
            else:
                print(f"    ⚠️ Chiplet-level {power_type} power data not available. Skipping average chiplet power plot.")

            # --- Plot 5: Combined System and Selected Chiplet Power Over Time ---
            if system_power_available and chiplet_power_available:
                chiplets_to_plot = [1, 10, 51]
                plottable_chiplet_power = {cid: chiplet_power[cid] for cid in chiplets_to_plot if cid in chiplet_power}

                if plottable_chiplet_power:
                    print(f"    📊 Generating combined system and selected chiplet power plot for {power_type} power...")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 2.9), gridspec_kw={'hspace': 0})

                    # --- Top Plot (ax1): Selected Chiplets ---
                    for chiplet_id, power_values in sorted(plottable_chiplet_power.items()):
                        if power_values is not None and len(power_values) == len(time_points):
                            ax1.plot(time_points, power_values, label=f'Chiplet {chiplet_id}', linewidth=1.5)
                    
                    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.0)
                        spine.set_color('black')
                    ax1.tick_params(axis='y', which='major', labelsize=8, length=3, width=0.8)
                    for label in ax1.get_yticklabels():
                        label.set_fontweight('bold')

                    # --- Bottom Plot (ax2): System Power ---
                    ax2.plot(time_points, system_power, label=f'System {title_power_type}', linewidth=1.5, linestyle='-', color='darkred')

                    ax2.set_xlabel('Time (μs)', fontweight='bold', fontsize=9, labelpad=2)
                    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
                    for spine in ax2.spines.values():
                        spine.set_linewidth(2.0)
                        spine.set_color('black')
                    ax2.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.8)
                    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                        label.set_fontweight('bold')

                    # --- Combined Figure Elements ---
                    fig.supylabel('Power (W)', fontweight='bold', fontsize=9, x=0.052)
                    
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    all_handles = handles1 + handles2
                    all_labels = labels1 + labels2
                    
                    # Make legend one line by setting ncol to the number of entries
                    legend = fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.52, 0.87), 
                                        ncol=len(all_handles) if len(all_handles) > 0 else 1, fontsize=9, frameon=True, borderaxespad=0.2)
                    if legend:
                        legend.get_frame().set_edgecolor('black')
                        legend.get_frame().set_linewidth(1.0)
                        for text in legend.get_texts():
                            text.set_fontweight('bold')
                    
                    # --- Save Plot ---
                    fig.tight_layout(rect=(0, 0, 1, 0.90))
                    plot_filename_base = f'combined_system_chiplet_{power_type}_power'
                    png_path = os.path.join(combined_power_dir, f"{plot_filename_base}.png")
                    pdf_path = os.path.join(combined_power_dir, f"{plot_filename_base}.pdf")
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    📊 Saved plots: {os.path.join('combined_power', plot_filename_base)}.png/pdf")

        print(f"\n📊 All power profile plots saved to: {power_plots_dir}")
        return power_plots_dir

    def _generate_comparison_plot(self, bar_data_list, network_types, title, ylabel, filename, plot_directory):
        """
        Helper function to generate a grouped bar chart for approach comparisons.

        Args:
            bar_data_list (list): List of tuples, e.g., [("Label1", data1_list), ("Label2", data2_list)]
            network_types (list): List of network type names for x-axis.
            title (str): Plot title.
            ylabel (str): Y-axis label.
            filename (str): Full path to save the plot (without extension).
            plot_directory (str): Directory where plot is saved (for print message).
        """
        num_network_types = len(network_types)
        if num_network_types == 0 or not bar_data_list:
            print(f"  ⚠️ No data to plot for {title}. Skipping.")
            return

        num_bars = len(bar_data_list)
        if num_bars == 0:
            print(f"  ⚠️ No bar data provided for {title}. Skipping.")
            return
            
        bar_group_total_width = 0.8 
        bar_width = bar_group_total_width / num_bars

        # Use a larger, publication-quality size
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        
        x = np.arange(num_network_types)
        
        # Use professional qualitative color palette for categorical data comparison
        # Best options: 'Set2' (clean/muted), 'Dark2' (high contrast), 'Paired' (good for comparisons)
        # Other good options: 'Set1', 'Accent', 'tab20', 'Pastel1', 'Pastel2'
        color_map = plt.get_cmap('Set2')
        colors = [color_map(i) for i in range(num_bars)]
        
        # Define patterns for each bar type (baseline gets no pattern)
        patterns = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', '...']
        
        # Define font sizes to match line chart
        label_fontsize = 9
        legend_fontsize = 9
        tick_fontsize = 8
        spine_linewidth = 2.0  # Bold outline
        
        multiplier = 0
        for idx, (label, data_values) in enumerate(bar_data_list):
            if len(data_values) != num_network_types:
                print(f"  ⚠️ Data length mismatch for '{label}' in '{title}'. Expected {num_network_types}, got {len(data_values)}. Skipping this bar.")
                pass # Assuming data_values will always be correctly sized from the caller

            offset = bar_width * multiplier
            
            # No pattern for baseline approach, patterns for others
            hatch_pattern = None if 'Baseline' in label else patterns[idx % len(patterns)]
            
            rects = ax.bar(x + offset, data_values, bar_width, label=label, color=colors[idx], 
                          edgecolor='black', linewidth=0.7, hatch=hatch_pattern)
            multiplier += 1

        ax.set_ylabel(ylabel, fontweight='bold', fontsize=label_fontsize, labelpad=2)
        # No plot title for publication
        # ax.set_title(title, fontweight='bold', fontsize=title_fontsize, pad=6)
        ax.set_xticks(x + bar_width * (num_bars - 1) / 2, network_types, rotation=15, ha="right")
        
        # Only show legend if more than one bar group
        if num_bars > 1:
            legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, borderaxespad=0.2, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            if legend:
                # Set legend frame color and line width to match line chart
                legend.get_frame().set_edgecolor('black')
                legend.get_frame().set_linewidth(1.0)
                legend.get_frame().set_alpha(0.95)
                for text in legend.get_texts():
                    text.set_fontweight('bold')
                if legend.get_title():
                    legend.get_title().set_fontweight('bold')
        
        ax.set_ylim(bottom=0)
        # Grid styling to match line chart
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7, linewidth=0.7)
        ax.xaxis.grid(False)

        # Spines styling (bold and black) to match line chart
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_color('black')

        # Tick labels styling (bold) to match line chart
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, length=3, width=0.8)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        try:
            plt.tight_layout(pad=0.5)
            # Save as both PNG and PDF with consistent sizing
            base_filename, _ = os.path.splitext(filename)
            png_filename = base_filename + '.png'
            pdf_filename = base_filename + '.pdf'
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
            print(f"  📊 Saved plot: {os.path.basename(png_filename)} and {os.path.basename(pdf_filename)} to {plot_directory}")
        except Exception as e:
            print(f"  ❌ Error saving plot {os.path.basename(filename)}: {e}")
        plt.close(fig)

    def plot_approach_comparison_metrics(self):
        """
        Plots comparisons of average execution time (total, compute, communication) 
        per input for each network type across different simulation approaches.
        Data is retrieved from the 'approach_comparison_metrics' attribute of MetricComputer.
        """
        print("\n📊 Generating approach comparison metrics plots...")

        if not self.metric_computer:
            print("  ⚠️ Error: MetricComputer instance not found.")
            return

        approach_metrics_data = getattr(self.metric_computer, 'approach_comparison_metrics', None)

        if not approach_metrics_data or not isinstance(approach_metrics_data, dict):
            print("  ⚠️ Approach comparison metrics not available or not a dict in MetricComputer. Skipping plots.")
            return
        
        comparison_data = approach_metrics_data.get('network_type_performance_comparison', None)

        if not comparison_data or not isinstance(comparison_data, dict) or not comparison_data:
            print("  ⚠️ 'network_type_performance_comparison' data is empty or not in the expected format. Skipping plots.")
            return

        network_types = sorted(comparison_data.keys())
        if not network_types:
            print("  ⚠️ No network types found in approach comparison metrics. Skipping plots.")
            return

        main_sim_total_times = []
        individual_model_total_times = []
        empty_system_total_times = []
        main_sim_compute_times = []
        main_sim_comm_times = []
        valid_network_types_for_plot = []
        empty_system_compute_times = []
        empty_system_comm_times = []

        for net_type in network_types:
            data = comparison_data[net_type]
            main_sim_data = data.get("main_simulation", {})
            individual_model_data = data.get("individual_model", {})
            empty_system_model_data = data.get("empty_system_model", {})

            main_total_avg = main_sim_data.get('avg_time_per_input')
            main_compute_avg = main_sim_data.get('avg_compute_time_per_input')
            main_comm_avg = main_sim_data.get('avg_comm_time_per_input')
            individual_total_avg = individual_model_data.get('avg_time_per_input')
            empty_total_avg = empty_system_model_data.get('avg_time_per_input')
            empty_compute_avg = empty_system_model_data.get('avg_compute_time_per_input')
            empty_comm_avg = empty_system_model_data.get('avg_comm_time_per_input')
            
            main_total_avg = main_total_avg if main_total_avg is not None else 0
            main_compute_avg = main_compute_avg if main_compute_avg is not None else 0
            main_comm_avg = main_comm_avg if main_comm_avg is not None else 0
            individual_total_avg = individual_total_avg if individual_total_avg is not None else 0
            empty_total_avg = empty_total_avg if empty_total_avg is not None else 0
            empty_compute_avg = empty_compute_avg if empty_compute_avg is not None else 0
            empty_comm_avg = empty_comm_avg if empty_comm_avg is not None else 0

            if main_total_avg > 0: 
                valid_network_types_for_plot.append(net_type)
                main_sim_total_times.append(main_total_avg)
                main_sim_compute_times.append(main_compute_avg)
                main_sim_comm_times.append(main_comm_avg)
                individual_model_total_times.append(individual_total_avg)
                empty_system_total_times.append(empty_total_avg)
                empty_system_compute_times.append(empty_compute_avg)
                empty_system_comm_times.append(empty_comm_avg)

        if not valid_network_types_for_plot:
            print("  ⚠️ No valid data to plot after filtering network types. Skipping all comparison plots.")
            return
            
        individual_model_compute_times = []
        individual_model_comm_times = []
        plot_individual_baseline = any(t > 0 for t in individual_model_total_times)

        if plot_individual_baseline:
            # For the individual model, total time is compute time, and comm time is zero.
            individual_model_compute_times = individual_model_total_times
            individual_model_comm_times = [0] * len(valid_network_types_for_plot)
        else: # If individual model has no data, create lists of zeros
            individual_model_compute_times = [0] * len(valid_network_types_for_plot)
            individual_model_comm_times = [0] * len(valid_network_types_for_plot)

        comparison_plots_dir = os.path.join(self.plots_folder, "approach_comparison_plots")
        os.makedirs(comparison_plots_dir, exist_ok=True)

        # --- Plot 1: Total Average Execution Time ---
        total_time_bars = [("Co-Simulation", main_sim_total_times)]
        if plot_individual_baseline:
            total_time_bars.append(("Individual Model", individual_model_total_times))
        total_time_bars.append(("Baseline (Total)", empty_system_total_times))
        total_time_bars.append(("Baseline (Comm)", empty_system_comm_times))
        self._generate_comparison_plot(
            bar_data_list=total_time_bars,
            network_types=valid_network_types_for_plot,
            title='Average Total Execution Time Comparison',
            ylabel='Avg. Model Latency (μs)',
            filename=os.path.join(comparison_plots_dir, "approach_total_runtime_comparison"),
            plot_directory=comparison_plots_dir
        )

        # --- Plot 2: Average Compute Time ---
        compute_time_bars = [("Co-Simulation", main_sim_compute_times)]
        if plot_individual_baseline:
             compute_time_bars.append(("Individual Model", individual_model_compute_times))
        compute_time_bars.append(("Baseline", empty_system_compute_times))
        self._generate_comparison_plot(
            bar_data_list=compute_time_bars,
            network_types=valid_network_types_for_plot,
            title='Average Compute Time Comparison',
            ylabel='Avg. Model Compute Latency (μs)',
            filename=os.path.join(comparison_plots_dir, "approach_compute_time_comparison"),
            plot_directory=comparison_plots_dir
        )

        # --- Plot 3: Average Communication Time ---
        comm_time_bars = [("Co-Simulation", main_sim_comm_times)]
        if plot_individual_baseline:
            comm_time_bars.append(("Individual Model", individual_model_comm_times))
        comm_time_bars.append(("Baseline", empty_system_comm_times))
        self._generate_comparison_plot(
            bar_data_list=comm_time_bars,
            network_types=valid_network_types_for_plot,
            title='Average Communication Time Comparison',
            ylabel='Avg. Model Comm. Latency (μs)',
            filename=os.path.join(comparison_plots_dir, "approach_comm_time_comparison"),
            plot_directory=comparison_plots_dir
        )

        # --- Plot 4: Stacked Bar plot of Compute and Communication time ---
        if any(t > 0 for t in main_sim_total_times) or any(t > 0 for t in empty_system_total_times):
            print("  📊 Generating stacked execution time comparison plot...")
            
            plt.style.use('seaborn-v0_8-whitegrid')
            networks = valid_network_types_for_plot
            
            # --- Data Setup ---
            co_sim_comm = np.array(main_sim_comm_times)
            co_sim_comp = np.array(main_sim_compute_times)
            base_total_comm = np.array(empty_system_comm_times)
            base_total_comp = np.array(empty_system_compute_times)
            ind_comm = np.array(individual_model_comm_times)
            ind_comp = np.array(individual_model_compute_times)
            base_comm_only_comp = np.array([0] * len(networks))

            x = np.arange(len(networks))
            
            # --- Bar and Figure Configuration ---
            color_map = plt.get_cmap('Set2')
            comp_hatch = '///'
            
            bar_definitions = []
            bar_definitions.append({"label": "Co-Simulation", "comm": co_sim_comm, "comp": co_sim_comp, "color": color_map(0)}) # Reverted color
            if plot_individual_baseline:
                bar_definitions.append({"label": "Individual Model", "comm": ind_comm, "comp": ind_comp, "color": color_map(1)})
            bar_definitions.append({"label": "Baseline (Comm. + Compute)", "comm": base_total_comm, "comp": base_total_comp, "color": 'white'})
            bar_definitions.append({"label": "Baseline (Comm. Only)", "comm": base_total_comm, "comp": base_comm_only_comp, "color": color_map(2)})

            num_groups = len(bar_definitions)
            bar_width = 0.8 / num_groups
            total_width = bar_width * num_groups
            
            offsets = np.arange(num_groups) * bar_width - (total_width / 2) + (bar_width / 2)

            num_networks = len(networks)
            fig_width = max(4.5, num_networks * (num_groups * 0.45)) # Make plot even narrower
            fig, ax = plt.subplots(figsize=(fig_width, 4))

            # --- Plot Bars ---
            for i, group in enumerate(bar_definitions):
                offset = offsets[i]
                ax.bar(x + offset, group["comm"], bar_width, color=group["color"], edgecolor='black', linewidth=1.1)
                ax.bar(x + offset, group["comp"], bar_width, bottom=group["comm"], color=group["color"], edgecolor='black', linewidth=1.1, hatch=comp_hatch)

            # --- Labels and Styling ---
            label_fontsize = 9
            legend_fontsize = 9
            tick_fontsize = 8
            spine_linewidth = 2.0

            ax.set_ylabel('Avg. Model Latency (μs)', fontweight='bold', fontsize=label_fontsize, labelpad=2)
            
            ax.set_xticks(x)
            ax.set_xticklabels([''] * len(networks))
            ax.tick_params(axis='x', length=0)

            # Add network names below the x-axis, closer to the plot
            y_offset = -0.05
            for i, net in enumerate(networks):
                ax.text(i, y_offset, net, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=tick_fontsize + 1, fontweight='bold')
            
            # --- Create a comprehensive legend with 3 rows ---
            # Bottom row for the texture representation
            texture_patches = [
                mpatches.Patch(facecolor='gray', label='Communication', edgecolor='black'),
                mpatches.Patch(facecolor='gray', hatch=comp_hatch, label='Compute', edgecolor='black')
            ]
            texture_legend = ax.legend(handles=texture_patches, loc='lower center', 
                                       bbox_to_anchor=(0.5, 1.02), 
                                       ncol=2, frameon=True, borderaxespad=0.2, 
                                       fontsize=legend_fontsize)
            if texture_legend:
                texture_legend.get_frame().set_edgecolor('black')
                texture_legend.get_frame().set_linewidth(1.0)
                for text in texture_legend.get_texts():
                    text.set_fontweight('bold')
            ax.add_artist(texture_legend)

            # Top rows for bar colors
            color_patches = []
            for group in bar_definitions:
                color_patches.append(mpatches.Patch(facecolor=group["color"], edgecolor='black', label=group["label"]))
            
            num_color_items = len(color_patches)
            color_legend_ncols = 2 if num_color_items > 2 else num_color_items

            color_legend = ax.legend(handles=color_patches, loc='lower center', 
                                     bbox_to_anchor=(0.5, 1.24), 
                                     ncol=color_legend_ncols, frameon=True, borderaxespad=0.2, 
                                     fontsize=legend_fontsize)
            if color_legend:
                color_legend.get_frame().set_edgecolor('black')
                color_legend.get_frame().set_linewidth(1.0)
                for text in color_legend.get_texts():
                    text.set_fontweight('bold')

            ax.set_ylim(bottom=0)
            plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.7)
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7, linewidth=0.7)

            for spine in ax.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_color('black')

            ax.tick_params(axis='y', which='major', labelsize=tick_fontsize, length=3, width=0.8)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            
            try:
                fig.tight_layout(rect=(0, 0.05, 1, 0.88)) # Adjust top/bottom margin for labels/legend
                filename = os.path.join(comparison_plots_dir, "approach_stacked_runtime_comparison")
                base_filename, _ = os.path.splitext(filename)
                png_filename = base_filename + '.png'
                pdf_filename = base_filename + '.pdf'
                plt.savefig(png_filename, dpi=300, bbox_inches='tight')
                plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
                print(f"  📊 Saved plot: {os.path.basename(png_filename)} and {os.path.basename(pdf_filename)} to {comparison_plots_dir}")
            except Exception as e:
                print(f"  ❌ Error saving stacked plot: {e}")
            plt.close(fig)

        print("📊 ...Finished generating approach comparison metrics plots.") 