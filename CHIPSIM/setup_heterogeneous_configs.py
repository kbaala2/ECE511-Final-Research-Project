"""
setup_heterogeneous_configs.py

Run this from the CHIPSIM directory:
    python3 setup_heterogeneous_configs.py

This script does three things:
1. Creates 6 heterogeneous chiplet mapping YAML files
2. Creates simulation config YAMLs for each mapping × 3 baseline models
3. Creates a clean organized results index with symlinks and a README
"""

import os
import yaml

# ============================================================
# PART 1: GENERATE HETEROGENEOUS MAPPING YAML FILES
# ============================================================

ROWS = 10
COLS = 10
MAPPING_DIR = "assets/chiplet_specs"
os.makedirs(MAPPING_DIR, exist_ok=True)

def get_chiplet_id(row, col):
    """Convert 0-indexed row/col to 1-indexed chiplet ID (row-major)."""
    return row * COLS + col + 1

def is_corner(row, col):
    return (row in [0, ROWS - 1]) and (col in [0, COLS - 1])

def generate_mapping(assignment_fn, filename, description):
    """
    assignment_fn(row, col) -> chiplet type string
    Corners are always IO regardless of assignment_fn.
    """
    mapping = {}
    for row in range(ROWS):
        for col in range(COLS):
            cid = get_chiplet_id(row, col)
            if is_corner(row, col):
                mapping[cid] = "IO"
            else:
                mapping[cid] = assignment_fn(row, col)

    path = os.path.join(MAPPING_DIR, filename)
    with open(path, "w") as f:
        f.write(f"# {description}\n")
        for k in sorted(mapping):
            f.write(f"{k}: {mapping[k]}\n")
    print(f"Created mapping: {path}")
    return path


# --- 1. Top-IMC / Bottom-CMOS (rename of existing heterogeneous) ---
def top_imc_bottom_cmos(row, col):
    return "IMC_E" if row < 5 else "CMOS_Compute"

generate_mapping(
    top_imc_bottom_cmos,
    "mapping_100_hetero_top_imc_bottom_cmos.yaml",
    "Heterogeneous: Top 5 rows IMC_E, Bottom 5 rows CMOS_Compute, IO at corners"
)

# --- 2. Left-IMC / Right-CMOS ---
def left_imc_right_cmos(row, col):
    return "IMC_E" if col < 5 else "CMOS_Compute"

generate_mapping(
    left_imc_right_cmos,
    "mapping_100_hetero_left_imc_right_cmos.yaml",
    "Heterogeneous: Left 5 columns IMC_E, Right 5 columns CMOS_Compute, IO at corners"
)

# --- 3. Center-IMC / Border-CMOS ---
# Inner 6x6 block (rows 2-7, cols 2-7) = IMC, everything else = CMOS
def center_imc_border_cmos(row, col):
    if 2 <= row <= 7 and 2 <= col <= 7:
        return "IMC_E"
    return "CMOS_Compute"

generate_mapping(
    center_imc_border_cmos,
    "mapping_100_hetero_center_imc.yaml",
    "Heterogeneous: IMC_E clustered in center 6x6 block, CMOS_Compute on outer ring, IO at corners"
)

# --- 4. Checkerboard ---
# (row+col) even = IMC, odd = CMOS
def checkerboard(row, col):
    return "IMC_E" if (row + col) % 2 == 0 else "CMOS_Compute"

generate_mapping(
    checkerboard,
    "mapping_100_hetero_checkerboard.yaml",
    "Heterogeneous: Alternating IMC_E and CMOS_Compute in checkerboard pattern, IO at corners"
)

# --- 5. IMC-Heavy (approx 75% IMC, 25% CMOS) ---
# Rows 0-6 = IMC (7 rows), Rows 7-9 = CMOS (3 rows)
def imc_heavy(row, col):
    return "IMC_E" if row < 7 else "CMOS_Compute"

generate_mapping(
    imc_heavy,
    "mapping_100_hetero_imc_heavy.yaml",
    "Heterogeneous: ~75% IMC_E (rows 1-7), ~25% CMOS_Compute (rows 8-10), IO at corners"
)

# --- 6. CMOS-Heavy (approx 25% IMC, 75% CMOS) ---
# Rows 0-2 = IMC (3 rows), Rows 3-9 = CMOS (7 rows)
def cmos_heavy(row, col):
    return "IMC_E" if row < 3 else "CMOS_Compute"

generate_mapping(
    cmos_heavy,
    "mapping_100_hetero_cmos_heavy.yaml",
    "Heterogeneous: ~25% IMC_E (rows 1-3), ~75% CMOS_Compute (rows 4-10), IO at corners"
)


# ============================================================
# PART 2: GENERATE SIMULATION CONFIGS FOR EACH MAPPING × MODEL
# ============================================================

CONFIG_DIR = "configs/experiments"
os.makedirs(CONFIG_DIR, exist_ok=True)

models = [
    ("alexnet",  "workload_alexnet.csv"),
    ("resnet50", "workload_resnet50.csv"),
    ("vgg16",    "workload_vgg16.csv"),
]

hetero_mappings = [
    ("hetero_top_imc_bottom_cmos", "mapping_100_hetero_top_imc_bottom_cmos.yaml"),
    ("hetero_left_imc_right_cmos", "mapping_100_hetero_left_imc_right_cmos.yaml"),
    ("hetero_center_imc",          "mapping_100_hetero_center_imc.yaml"),
    ("hetero_checkerboard",        "mapping_100_hetero_checkerboard.yaml"),
    ("hetero_imc_heavy",           "mapping_100_hetero_imc_heavy.yaml"),
    ("hetero_cmos_heavy",          "mapping_100_hetero_cmos_heavy.yaml"),
]

def make_config(workload, mapping_file):
    return {
        "simulation": {
            "input_files": {
                "workload": workload,
                "adj_matrix": "adj_matrix_10x10_mesh.csv",
                "chiplet_mapping": mapping_file,
                "model_defs": "model_definitions.py",
            },
            "core_settings": {
                "clear_cache": False,
                "comm_simulator": "Garnet",
                "comm_method": "non-pipelined",
                "enable_dsent": False,
                "enable_comm_cache": True,
                "blocking_age_threshold": 10,
                "weight_stationary": True,
                "weight_loading_strategy": "all_at_once",
            },
            "hardware_parameters": {
                "bits_per_activation": 8,
                "bits_per_packet": 128,
                "network_operation_frequency_hz": 1000000000,
            },
            "gem5_parameters": {
                "gem5_sim_cycles": 500000000,
                "gem5_injection_rate": 0.0,
                "gem5_ticks_per_cycle": 1000,
                "gem5_deadlock_threshold": None,
            },
            "dsent_parameters": {
                "dsent_tech_node": "32",
            },
        },
        "post_processing": {
            "warmup_period_us": 0.0,
            "cooldown_period_us": 0.0,
            "run_wkld_agg_comm": False,
            "run_ind_comm": False,
            "run_net_agg_comm": False,
            "generate_plots": True,
            "generate_visualizations": False,
        },
    }

hetero_config_names = []
for model_name, workload_file in models:
    for mapping_name, mapping_file in hetero_mappings:
        config_name = f"{model_name}_{mapping_name}"
        cfg = make_config(workload_file, mapping_file)
        path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        hetero_config_names.append(config_name)
        print(f"Created config: {path}")


# ============================================================
# PART 3: CREATE CLEAN ORGANIZED RESULTS INDEX
# ============================================================

# Map clean names to actual result folder names (from April 6th canonical baseline runs)
RESULTS_BASE = "_results/formatted_results"
BENCHMARKS_DIR = "performance_benchmarks"

canonical_results = {
    "baseline_alexnet_cmos":   "2026.04.06_17.00.47_workload_alexnet_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_cmos_with_io_100chiplets",
    "baseline_alexnet_imc":    "2026.04.06_17.21.07_workload_alexnet_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_with_io_100chiplets",
    "baseline_resnet50_cmos":  "2026.04.06_17.23.53_workload_resnet50_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_cmos_with_io_100chiplets",
    "baseline_resnet50_imc":   "2026.04.06_17.35.31_workload_resnet50_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_with_io_100chiplets",
    "baseline_vgg16_cmos":     "2026.04.06_17.38.20_workload_vgg16_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_cmos_with_io_100chiplets",
    "baseline_vgg16_imc":      "2026.04.06_17.44.37_workload_vgg16_Garnet_non-pipelined_adj_matrix_10x10_mesh_mapping_100_with_io_100chiplets",
}

# Create organized benchmark directory with subdirectories for baseline and heterogeneous
os.makedirs(os.path.join(BENCHMARKS_DIR, "baseline"), exist_ok=True)
os.makedirs(os.path.join(BENCHMARKS_DIR, "heterogeneous"), exist_ok=True)

# Create symlinks for baseline results
for clean_name, actual_folder in canonical_results.items():
    src = os.path.abspath(os.path.join(RESULTS_BASE, actual_folder))
    dst = os.path.join(BENCHMARKS_DIR, "baseline", clean_name)
    if os.path.exists(src):
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"Symlink: {dst} -> {src}")
    else:
        print(f"WARNING: Source folder not found: {src}")

# Write a README index
readme_path = os.path.join(BENCHMARKS_DIR, "README.md")
with open(readme_path, "w") as f:
    f.write("# AutoMap Performance Benchmarks\n\n")
    f.write("## Directory Structure\n\n")
    f.write("```\n")
    f.write("performance_benchmarks/\n")
    f.write("├── baseline/          # Nearest Neighbor mapping results (CMOS and IMC)\n")
    f.write("└── heterogeneous/     # Results from heterogeneous chiplet configurations\n")
    f.write("```\n\n")

    f.write("## Baseline Results\n\n")
    f.write("All baseline runs use Nearest Neighbor mapping, non-pipelined communication,\n")
    f.write("all-at-once weight loading, weight-stationary dataflow, on a 10x10 mesh topology.\n\n")
    f.write("| Clean Name | Model | Chiplet Config | Latency (μs) | Actual Folder |\n")
    f.write("|---|---|---|---|---|\n")

    latencies = {
        "baseline_alexnet_cmos":  14225.92,
        "baseline_alexnet_imc":   13541.81,
        "baseline_resnet50_cmos": 13015.37,
        "baseline_resnet50_imc":  5518.65,
        "baseline_vgg16_cmos":    35014.59,
        "baseline_vgg16_imc":     20471.04,
    }

    for clean_name, actual_folder in canonical_results.items():
        parts = clean_name.replace("baseline_", "").split("_")
        chiplet = parts[-1].upper()
        model = "_".join(parts[:-1])
        latency = latencies.get(clean_name, "N/A")
        f.write(f"| {clean_name} | {model} | {chiplet} | {latency} | {actual_folder} |\n")

    f.write("\n## Heterogeneous Configurations\n\n")
    f.write("| Config File | IMC Chiplets | CMOS Chiplets | IO Chiplets | Layout Description |\n")
    f.write("|---|---|---|---|---|\n")
    f.write("| mapping_100_hetero_top_imc_bottom_cmos.yaml | 46 | 50 | 4 | Top 5 rows IMC, bottom 5 rows CMOS |\n")
    f.write("| mapping_100_hetero_left_imc_right_cmos.yaml | 46 | 50 | 4 | Left 5 columns IMC, right 5 columns CMOS |\n")
    f.write("| mapping_100_hetero_center_imc.yaml | 36 | 60 | 4 | IMC in center 6x6 block, CMOS on outer ring |\n")
    f.write("| mapping_100_hetero_checkerboard.yaml | 48 | 48 | 4 | Alternating IMC/CMOS checkerboard pattern |\n")
    f.write("| mapping_100_hetero_imc_heavy.yaml | 66 | 30 | 4 | ~75% IMC (rows 1-7), ~25% CMOS (rows 8-10) |\n")
    f.write("| mapping_100_hetero_cmos_heavy.yaml | 26 | 70 | 4 | ~25% IMC (rows 1-3), ~75% CMOS (rows 4-10) |\n")

    f.write("\n## Simulation Commands\n\n")
    f.write("### Run all heterogeneous configs:\n")
    f.write("```bash\n")
    f.write("for config in \\\n")
    for name in hetero_config_names:
        f.write(f"  {name} \\\n")
    f.write("; do\n")
    f.write("    python3 simulate.py --mode simulate --config $config\n")
    f.write("done\n")
    f.write("```\n")
    f.write("\n> **Note:** All heterogeneous configs include IMC chiplets and require\n")
    f.write("> the CIMLoop Docker container to be running before starting simulations.\n")
    f.write("> Start it with: `docker start cimloop-api && docker exec -it cimloop-api bash -c 'cd /home/api_server && python3 api_server.py'`\n")

print(f"\nCreated README: {readme_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n6 heterogeneous mapping YAMLs created in {MAPPING_DIR}/")
print(f"18 simulation configs created in {CONFIG_DIR}/")
print(f"  (6 mappings × 3 models: alexnet, resnet50, vgg16)")
print(f"Clean benchmark directory created at {BENCHMARKS_DIR}/")
print(f"  - baseline/ contains symlinks to 6 canonical baseline results")
print(f"  - heterogeneous/ ready to receive new results")
print(f"  - README.md index documents everything")
print(f"\nNext steps:")
print(f"1. Make sure CIMLoop Docker container is running (IMC chiplets needed)")
print(f"2. Run heterogeneous simulations:")
print(f"   for config in {' '.join(hetero_config_names[:3])} ...; do")
print(f"       python3 simulate.py --mode simulate --config $config")
print(f"   done")
print(f"3. Push performance_benchmarks/ to GitHub")
