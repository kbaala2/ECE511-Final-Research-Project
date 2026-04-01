# class_OutputManager.py

import os
import time

class OutputManager:
    """
    Manages simulation output formatting, printing, and saving to files.
    Handles outputs for the main simulation, aggregate communication, and individual layer communication.
    Accepts a MetricComputer instance to retrieve pre-computed metrics.
    
    Args:
        wl_file_name (str): Name of the workload file
        adj_matrix_file (str): Name of the adjacency matrix file
        chiplet_mapping_file (str): Name of the chiplet mapping file
        communication_simulator (str): Name of the communication simulator used
        communication_method (str): Method for communication ("pipelined" or "non-pipelined")
        mapping_function (str): Name of the mapping function used
        metric_computer (MetricComputer): Instance containing computed simulation metrics
        results_dir (str, optional): Path to the directory where results will be saved. 
                                    If None, a unique directory will be created.
    """

    def __init__(
        self,
        wl_file_name,
        adj_matrix_file,
        chiplet_mapping_file,
        communication_simulator,
        communication_method,
        mapping_function,
        metric_computer, # Added metric_computer
        results_dir=None,
        num_chiplets=None
    ):
        self.wl_file_name = wl_file_name
        self.adj_matrix_file = adj_matrix_file
        self.chiplet_mapping_file = chiplet_mapping_file
        self.communication_simulator = communication_simulator
        self.communication_method = communication_method
        self.mapping_function = mapping_function
        self.metric_computer = metric_computer # Store the metric computer instance
        self.results_dir = results_dir
        self.num_chiplets = num_chiplets
        
        # Store the base name of the workload file (without extension) for file naming
        self.workload_base_name = os.path.splitext(os.path.basename(wl_file_name))[0]
        
    def create_results_directory(self):
        """
        Create a unique directory for storing raw simulation results under _results/raw_results/.
        
        Returns:
            str: Path to the created results directory (e.g., _results/raw_results/YYYY.MM.DD_..._params)
        """
        # Create base _results directory if it doesn't exist
        base_results_root = os.path.join(os.getcwd(), "_results")
        os.makedirs(base_results_root, exist_ok=True)

        # Create _results/raw_results/ directory if it doesn't exist
        raw_results_base_dir = os.path.join(base_results_root, "raw_results")
        os.makedirs(raw_results_base_dir, exist_ok=True)
        
        # Create a unique directory name based on timestamp and parameters
        timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")
        
        adj_matrix_base_name = os.path.splitext(os.path.basename(self.adj_matrix_file))[0]
        chiplet_mapping_base_name = os.path.splitext(os.path.basename(self.chiplet_mapping_file))[0]

        # Include relevant simulation parameters in the directory name
        dir_name = (f"{timestamp}_{self.workload_base_name}_{self.communication_simulator}_"
                    f"{self.communication_method}_{adj_matrix_base_name}_"
                    f"{chiplet_mapping_base_name}_{self.num_chiplets}chiplets")
        
        # Create the unique directory path under raw_results_base_dir
        self.results_dir = os.path.join(raw_results_base_dir, dir_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # print(f"✅ Created unique raw results directory: {self.results_dir}")
        return self.results_dir
    

    def save_formatted_metrics(self, metric_root, subdirectory=None):
        """
        Save formatted model metrics from MetricFormatter to files (now using unified MetricNode format).
        The behavior depends on metric_root.save_children_separately:
        - If False: Saves metric_root as a single file with its children included.
        - If True: Treats metric_root as a container, saving each of its children as separate files/entities.
        Args:
            metric_root (MetricNode): The root MetricNode containing metrics.
            subdirectory (str, optional): Subdirectory relative to self.results_dir to save the metrics to.
                                        If None, saves to top-level results directory.
        Returns:
            str: Path to the main directory where metrics were saved for this call.
        """
        # Determine base output directory for this save operation
        if subdirectory is None:
            main_output_dir = self.results_dir
        else:
            subdirectory = metric_root.children_folder_name if subdirectory is None else subdirectory
            main_output_dir = os.path.join(self.results_dir, subdirectory)
            os.makedirs(main_output_dir, exist_ok=True)

        if not metric_root.save_children_separately:
            # Case: metric_root is a single report (e.g., Utilization Metrics).
            # Its children are part of this single report.
            safe_name = metric_root.name.replace(":", "_").replace("/", "_").replace(" ", "_")
            file_path = os.path.join(main_output_dir, f"{safe_name}.txt")
            with open(file_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"{metric_root.name}\n")
                f.write("=" * 80 + "\n\n")
                self._write_metric_node_content_to_file(metric_root, f, indent=0)
            # print(f"✅ Report '{metric_root.name}' saved to {file_path}")
        else:
            # Case: metric_root's children should be separate files/entities (e.g., "All Models" root).
            # Create an index file in main_output_dir for the children of metric_root.
            index_file_path = os.path.join(main_output_dir, "index.txt")
            with open(index_file_path, 'w') as f_index:
                f_index.write("=" * 80 + "\n")
                f_index.write(f"INDEX: {metric_root.name}\n")
                f_index.write("=" * 80 + "\n\n")
                f_index.write(f"Workload: {self.wl_file_name}\n")
                f_index.write(f"Communication Simulator: {self.communication_simulator}\n")
                f_index.write(f"Communication Method: {self.communication_method}\n\n")
                f_index.write("Contents (files/subdirectories listed below correspond to items in this directory):\n")
                if not metric_root.children:
                    f_index.write("  No child metrics found.\n")
                else:
                    for child_node in metric_root.children:
                        f_index.write(f"  - {child_node.name}")
                        if child_node.children_folder_name:
                             f_index.write(f" (see subdirectory: {child_node.children_folder_name})")
                        f_index.write("\n")
            # print(f"✅ Index for '{metric_root.name}' saved to {index_file_path}")

            # Process each child of metric_root
            for child_node in metric_root.children:
                current_child_output_dir = main_output_dir
                if child_node.children_folder_name:
                    # This child wants its own sub-folder relative to self.results_dir
                    current_child_output_dir = os.path.join(self.results_dir, child_node.children_folder_name)
                    os.makedirs(current_child_output_dir, exist_ok=True)

                child_safe_name = child_node.name.replace(":", "_").replace("/", "_").replace(" ", "_")
                child_main_file_path = os.path.join(current_child_output_dir, f"{child_safe_name}.txt")

                if child_node.save_children_separately:
                    # This child_node (e.g., a specific Model from "All Models") ALSO wants its own
                    # children (e.g., sections within that model) to be in separate files.
                    # This implies child_node acts as a mini-root for its own content.
                    with open(child_main_file_path, 'w') as f_child_main:
                        f_child_main.write("=" * 80 + "\n")
                        f_child_main.write(f"{child_node.name}\n")
                        f_child_main.write("=" * 80 + "\n\n")
                        # Write ONLY the child_node's direct content. Its children will be separate files.
                        # _write_metric_node_content_to_file respects child_node.save_children_separately.
                        self._write_metric_node_content_to_file(child_node, f_child_main, indent=0)
                    
                    child_sub_index_path = os.path.join(current_child_output_dir, "index.txt") \
                        if child_node.children_folder_name else \
                        os.path.join(current_child_output_dir, f"{child_safe_name}_index.txt")

                    with open(child_sub_index_path, 'w') as f_child_sub_index:
                        f_child_sub_index.write("=" * 80 + "\n")
                        f_child_sub_index.write(f"INDEX: {child_node.name}\n")
                        f_child_sub_index.write("=" * 80 + "\n\n")
                        f_child_sub_index.write(f"Contents of '{child_node.name}':\n")
                        if not child_node.children:
                             f_child_sub_index.write("  No sub-metrics found.\n")
                        else:
                            for grandchild_node in child_node.children:
                                f_child_sub_index.write(f"  - {grandchild_node.name}\n")
                    
                    for grandchild_node in child_node.children:
                        grandchild_safe_name = grandchild_node.name.replace(":", "_").replace("/", "_").replace(" ", "_")
                        grandchild_file_path = os.path.join(current_child_output_dir, f"{grandchild_safe_name}.txt")
                        with open(grandchild_file_path, 'w') as f_grandchild:
                            f_grandchild.write("=" * 80 + "\n")
                            f_grandchild.write(f"{grandchild_node.name}\n")
                            f_grandchild.write("=" * 80 + "\n\n")
                            self._write_metric_node_content_to_file(grandchild_node, f_grandchild, indent=0)
                else:
                    # child_node.save_children_separately is False (standard case for a model metric node).
                    # Save this child_node and all its children (recursively) into child_main_file_path.
                    with open(child_main_file_path, 'w') as f_child_main:
                        f_child_main.write("=" * 80 + "\n")
                        f_child_main.write(f"{child_node.name}\n")
                        f_child_main.write("=" * 80 + "\n\n")
                        self._write_metric_node_content_to_file(child_node, f_child_main, indent=0)
        
        return main_output_dir

    # Helper to recursively write a MetricNode to file
    def _write_metric_node_content_to_file(self, node, file, indent=0, print_section_header=True): # Renamed from write_metric_node
        prefix = "  " * indent
        # Group children by section for consecutive printing
        def group_by_section(children):
            groups = []
            if not children:
                return groups
            current_section = children[0].section
            current_group = []
            for child in children:
                if child.section == current_section:
                    current_group.append(child)
                else:
                    groups.append((current_section, current_group))
                    current_section = child.section
                    current_group = [child]
            if current_group:
                groups.append((current_section, current_group))
            return groups

        # Only print the section header if allowed
        if node.section and print_section_header and (indent == 0 or node.section != node.name) and not node.children:
            file.write(f"{prefix}{node.section}:\n")
            file.write(f"{prefix}{'-'*50}\n")
        # Name and value (simple metric)
        if node.value is not None and (node.values is None):
            formatted_value_str = ""
            if isinstance(node.value, (int, float)) and node.fmt and node.unit:
                formatted_value_str = f"{node.value:{node.fmt}} {node.unit}"
            elif isinstance(node.value, (int, float)) and node.fmt:
                formatted_value_str = f"{node.value:{node.fmt}}"
            else:
                formatted_value_str = f"{node.value} {node.unit}"

            if node.section == "Model Mapping" and node.name == "Total chiplets used":
                file.write(f"{prefix}--- {node.name} ---\n")
                file.write(f"{prefix}  Value: {formatted_value_str}\n")
                file.write(f"{prefix}{'-' * (len(node.name) + 8)}\n\n") # Adjust separator length
            else:
                file.write(f"{prefix}{node.name}: {formatted_value_str}\n")
        # Structured values (dict)
        if node.values is not None:
            file.write(f"{prefix}{node.name}:\n")
            for k, v in node.values.items():
                if isinstance(v, (int, float)) and v >= 0:
                    file.write(f"{prefix}  {k}: {v:{node.fmt}} {node.unit}\n")
                elif isinstance(v, (int, float)) and v < 0:
                    file.write(f"{prefix}  {k}: N/A\n") # Handle negative values as N/A
                else:
                    file.write(f"{prefix}  {k}: {v} {node.unit}\n")
        # Table formatting (general)
        elif node.columns and node.rows:
            # Calculate column widths based on header and data
            col_widths = {col: len(str(col)) for col in node.columns}
            for row in node.rows:
                for col_name in node.columns:
                    value_str = str(row.get(col_name, ""))
                    col_widths[col_name] = max(col_widths[col_name], len(value_str))

            file.write(f"{prefix}{node.name}:\n\n") # Print table name

            # Top border
            border_line = prefix + "+-" + "-+-".join(["-" * col_widths[col] for col in node.columns]) + "-+"
            file.write(f"{border_line}\n")

            # Header row
            header_parts = [f"{str(col):<{col_widths[col]}}" for col in node.columns]
            file.write(f"{prefix}| " + " | ".join(header_parts) + " |\n")

            # Middle border (separator after header)
            file.write(f"{border_line}\n")

            # Data rows
            for row in node.rows:
                row_parts = []
                for col_name in node.columns:
                    value = row.get(col_name, "")
                    width = col_widths[col_name]
                    if isinstance(value, float):
                        formatted_val = f"{value:<{width}.2f}"
                    elif isinstance(value, int) and not isinstance(value, bool):
                        formatted_val = f"{value:<{width}}"
                    else:
                        formatted_val = f"{str(value):<{width}}"
                    row_parts.append(formatted_val)
                file.write(f"{prefix}| " + " | ".join(row_parts) + " |\n")

            # Bottom border
            file.write(f"{border_line}\n")
        # Description (for empty tables or explanations)
        if node.description:
            # If it's an empty table, it might have columns but no rows.
            # Avoid printing description if table was already printed.
            if not (node.columns and node.rows):
                 file.write(f"{prefix}{node.description}\n")

        # Children (nested metrics), grouped by section
        if node.children and not node.save_children_separately:
            for section, group in group_by_section(node.children):
                if section:
                    # Print section header only once for the group
                    file.write(f"\n{prefix}{section}:\n{prefix}{'-'*50}\n")
                for child in group:
                    # Don't print section header again in child
                    self._write_metric_node_content_to_file(child, file, indent=indent+1, print_section_header=False) # Renamed self.write_metric_node