# Chiplet System Simulator

A co-simulation framework for evaluating DNN workload execution on chiplet-based in-memory computing systems. The simulator models both computation (analog in-memory compute) and communication (Network-on-Interposer) with cycle-accurate network simulation.

## NOTE
- **Simulator is in its initial release, full correct functionality not yet guaranteed.**

- Thermal integration is currently manual


## Paper

CHIPSIM paper found here: [10.1109/OJSSCS.2025.3626314](https://doi.org/10.1109/OJSSCS.2025.3626314)

## Documentation

- **[Components & Architecture](docs/components.md)** - Core modules and system architecture
- **[Advanced Features](docs/advanced-features.md)** - Communication caching, DSENT, weight strategies
- **[GEM5 Integration](docs/gem5-integration.md)** - Network simulator setup and usage
- **[Helper Scripts](docs/helper-scripts.md)** - Workload and topology generation utilities

## Quick Start

1. **Run a simulation**
   ```bash
   python3 simulate.py --mode simulate --config <config_name>
   ```

2. **Re-process existing results**
   ```bash
   python3 simulate.py --mode reprocess --config <config_name> --results-dir <raw_results_dir>
   ```

3. **View outputs**
   - Raw results: `_results/raw_results/<timestamp>_<summary>/`
   - Formatted results: `_results/formatted_results/<raw_results_dir_name>/`
   - Metrics, plots, and visualizations included

## Prerequisites

- **Python 3.7+**: The simulator requires Python 3.7 or higher
- **CIMLoop API container**: IMC chiplet compute relies on `CIMLoopBackend` calling the REST API defined in `integrations/CIMLoop_API.py`, which targets `http://localhost:5000` by default. You must run the modified CIMLoop Docker container that exposes this API endpoint. Without the container the simulator cannot obtain latency and energy for IMC chiplets and the run will abort when those chiplets are scheduled. CMOS chiplets use the analytical compute backend and can still simulate without CIMLoop.

  **Docker setup (required for IMC simulation)**
  ```bash
  docker pull pfromm/cimloop-api:0.1.1
  docker run -d --name cimloop-api -p 5000:5000 pfromm/cimloop-api:0.1.1
  ```

  **Start the CIMLoop API server (required)**
  Open a second terminal and run:
  ```bash
  docker exec -it cimloop-api bash
  cd /home/api_server
  python3 api_server.py
  ```
  Keep this terminal open while running the simulator (it hosts the API on `http://localhost:5000`).

  Notes:
  - Use `pfromm/cimloop-api:0.1.1` or newer (it includes the required `/home/workspace` components in the image).
  - Do **not** mount a host directory onto `/home/workspace` unless you know what you are doing (it can hide Python modules the API server imports).
  - If you want to persist API outputs, mount only the outputs subfolder:
    ```bash
    mkdir -p cimloop_outputs
    docker run -d --name cimloop-api -p 5000:5000 \
      -v "$PWD/cimloop_outputs:/home/workspace/outputs" \
      pfromm/cimloop-api:0.1.1
    ```
  - The image also contains a Jupyter environment (port 8888), but it is not required for simulator integration.
  - This container is derived from the upstream CiMLoop project described in [CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool (ISPASS 2024)](https://arxiv.org/abs/2405.07259).

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r /docs/requirements.txt
   ```
   
   Required packages:
   - `numpy` - Numerical operations
   - `scipy` - Scientific computing (thermal model)
   - `networkx` - Graph operations for topology
   - `pyyaml` - Configuration file parsing
   - `matplotlib` - Plotting and visualization
   - `requests` - CIMLoop API communication
   - `scons` - Build system for gem5 compilation

2. **External tools** (optional, depending on your configuration):
   - **CIMLoop**: For IMC chiplet compute simulation
   - **gem5**: For detailed network simulation (if not using the simple model)
   - **DSENT**: For interconnect power/energy modeling

## Simulation Workflow

```
Config YAML → Run Simulation → Post-Process Results
     ↓              ↓                    ↓
  Setup params   Execute DNN       Analyze metrics
  Input files    Compute + Comms   Generate plots
```

**Main stages:**
1. **Configuration**: Define workload, topology, chiplet mapping, and simulation parameters
2. **Simulation**: Execute workload with compute and network co-simulation
3. **Post-processing**: Analyze results, compute metrics, generate visualizations

## Input Files

All input files are located in `assets/`:

### Workload Files (`assets/workloads/*.csv`)
Defines DNN inference requests and their injection times.

**Format:**
```csv
net_idx,inject_time_us,network,num_inputs
1,0,alexnet,1
2,100,resnet50,2
```

- `net_idx`: Unique network instance ID
- `inject_time_us`: When to inject the inference request
- `network`: DNN model name (must exist in model definitions)
- `num_inputs`: Batch size for this inference

**Generate custom workloads:**
```bash
python3 helpers/helpers_GenerateWorkload.py
```
See [Helper Scripts](docs/helper-scripts.md) for details.

### Network Topology (`assets/NoI_topologies/*.csv`)
Adjacency matrix defining the Network-on-Interposer (NoI) interconnect topology.

**Format:** CSV adjacency matrix where `matrix[i][j] = 1` indicates a direct link from chiplet `i` to chiplet `j`.

When Garnet runs in AnyNET mode, the simulator converts the selected adjacency matrix into `integrations/gem5/configs/topologies/myTopology.yaml`. Links that stay within a row (east/west) keep weight `1`, while links that move between rows (north/south) are assigned weight `2` to enforce XY-style ordering. Mixed or diagonal connections default to weight `1`.

**Common topologies:**
- `adj_matrix_10x10_mesh.csv`: 100-chiplet 2D mesh
- `adj_matrix_10x10_floret.csv`: Floret topology
- `adj_matrix_10x10_mesh_with_io.csv`: Mesh with I/O chiplets

**Generate custom size mesh topologies:**
```bash
python3 helpers/helpers_GenerateAdjMatrix.py
```
See [Helper Scripts](docs/helper-scripts.md) for details. For more involved layouts, you can supply your own adjacency matrix or edit the generated YAML inside `integrations/gem5/configs/topologies/`; see [GEM5 Integration](docs/gem5-integration.md#anynet-topology-conversion) for guidance.

### Chiplet Mapping (`assets/chiplet_specs/*.yaml`)
Maps chiplet IDs to chiplet types and their compute specifications.

**Format:**
```yaml
1: IO
2: IMC_A
3: IMC_B
...
```

**Available chiplet types (examples):** `IMC_A`, `IMC_B`, `IMC_C`, `IMC_D`, `IMC_E`, `IMC_F`, `IO`, `CMOS_Compute`
- Each type defined in `assets/chiplet_specs/chiplet_params.py`
- Types differ in compute capabilities, area, and power
- Memory model:
  - IMC chiplets: capacity tracked as crossbars; available memory derives from available crossbars × memory per crossbar
  - CMOS chiplets: capacity tracked as weight units; `total_memory_weights` is defined in params and decrements when weights are mapped

### Model Definitions (`assets/DNN_models/model_definitions.py`)
Python dictionary defining DNN architectures layer-by-layer.

**Structure:**
```python
MODEL_DEFINITIONS = {
    'alexnet': {
        'layers': {
            0: {
                'description': 'Conv2d layer',
                'parameters': { ... },
                'receiving_layers': [1]
            },
            ...
        }
    }
}
```

## Configuration

Simulation configs are YAML files in `configs/experiments/`. Each config specifies all simulation parameters.

### Example Config (`configs/experiments/config_1.yaml`)

```yaml
simulation:
  # Input files
  input_files:
    workload: "workload_alexnet.csv"
    adj_matrix: "adj_matrix_10x10_mesh.csv"
    chiplet_mapping: "mapping_100_with_io.yaml"  # or "mapping_100_cmos_with_io.yaml" for CMOS-only compute
    model_defs: "model_definitions.py"

  # Core simulation settings
  core_settings:
    clear_cache: false
    comm_simulator: "Garnet"           # 'Garnet' or 'Booksim' (Booksim not implemented)
    comm_method: "pipelined"            # 'pipelined' or 'non-pipelined'
    enable_dsent: false
    enable_comm_cache: true
    warmup_period_us: 0.0
    blocking_age_threshold: 10
    weight_stationary: true
    weight_loading_strategy: "all_at_once" # or "just_in_time"

  # Network / hardware parameters
  hardware_parameters:
    bits_per_activation: 8
    bits_per_packet: 128
    network_operation_frequency_hz: 1000000000

  # gem5-specific parameters
  gem5_parameters:
    gem5_sim_cycles: 500000000
    gem5_injection_rate: 0.0
    gem5_ticks_per_cycle: 1000
    gem5_deadlock_threshold: null

  # DSENT parameters
  dsent_parameters:
    dsent_tech_node: "32"

# Post-processing configuration (optional; enables auto post-processing)
post_processing:
  warmup_period_us: 0.0
  cooldown_period_us: 0.0
  run_wkld_agg_comm: false
  run_ind_comm: false
  run_net_agg_comm: false
  generate_plots: true
  generate_visualizations: false
```

### Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `comm_simulator` | `Garnet`, `Booksim` | Only Garnet supported (Booksim not implemented) |
| `comm_method` | `pipelined`, `non-pipelined` | Communication scheduling approach |
| `weight_stationary` | `true`, `false` | Weights loaded once vs. per-inference |
| `weight_loading_strategy` | `all_at_once`, `just_in_time` | Weight loading timing |
| `enable_dsent` | `true`, `false` | Enable power estimation |

See [Advanced Features](docs/advanced-features.md) for detailed parameter descriptions and [GEM5 Integration](docs/gem5-integration.md) for network simulation setup.

## Running Simulations

### Basic Usage

```bash
python3 simulate.py --mode simulate --config config_1
```

The config name refers to files in `configs/experiments/` (e.g., `config_1` → `configs/experiments/config_1.yaml`).

Other modes:
```bash
# Re-process a past run into formatted results
python3 simulate.py --mode reprocess --config config_1 --results-dir <raw_results_dir>

# Cross-simulation analysis (see configs/cross_analysis/*.yaml)
python3 simulate.py --mode cross-analysis --config example_comparison
```

### What Happens During Simulation

1. **Load configuration** and input files
2. **Build system model**: Create chiplets and NoI topology
3. **Map DNN layers** to chiplets using partitioning algorithm
4. **Execute workload**:
   - Compute simulation: Model analog compute latency
   - Communication simulation: gem5 Garnet for NoI
   - Weight scheduling: Manage weight transfers
5. **Save results**: Pickled state and raw metrics to `_results/`

See [Components & Architecture](docs/components.md) for detailed information about the simulation engine.

### Simulation Output

Results are saved to `_results/raw_results/<timestamp>/`:
- `simulation_state.pkl`: Complete simulation state
- `simulation_config.yaml`: Config used for the run
- Raw timing and communication logs

## Post-Processing

If your config includes a `post_processing` section, post-processing runs automatically.

To re-run post-processing later:
```bash
python3 simulate.py --mode reprocess --config <config_name> --results-dir <raw_results_dir>
```

### Analysis Features

Configure per-simulation in job files:
- **Warmup filtering**: Exclude initial warmup period from metrics
- **Communication analysis**: Per-workload, per-network, or aggregate
- **Visualization**: Chiplet utilization heatmaps, network traffic
- **Plots**: Timeline plots, throughput analysis

### Output

Enhanced results added to `_results/formatted_results/<raw_results_dir_name>/`:
- `metrics_summary.txt`: Key performance metrics
- `plots/`: Timeline and communication plots
- `visualizations/`: Chiplet mapping and utilization
- `cross_sim_comparison.csv`: Multi-run comparisons

## Additional Resources

- **[Components & Architecture](docs/components.md)** - Detailed module descriptions and directory structure
- **[Advanced Features](docs/advanced-features.md)** - Communication caching, weight strategies, DSENT
- **[GEM5 Integration](docs/gem5-integration.md)** - Installation, testing, and power analysis
- **[Helper Scripts](docs/helper-scripts.md)** - Input file generation and model management