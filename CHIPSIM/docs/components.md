# Major Components

This document describes the core modules and architecture of the Chiplet System Simulator.

## Core Modules (`src/`)

| Module | Description |
|--------|-------------|
| [`sim/global_manager.py`](../src/sim/global_manager.py) | Main simulation orchestrator |
| [`sim/compute_simulation/compute_simulator.py`](../src/sim/compute_simulation/compute_simulator.py) | Analog compute modeling |
| [`sim/communication_simulator.py`](../src/sim/communication_simulator.py) | NoI traffic simulation (gem5 Garnet) |
| [`core/system.py`](../src/core/system.py) | Chiplet system representation |
| [`mapping/model_mapper.py`](../src/mapping/model_mapper.py) | DNN-to-chiplet mapping algorithm |
| [`post/metrics/`](../src/post/metrics/) | Metrics computation and formatting |
| [`post/`](../src/post/) | Post-processing and visualization |

## Architecture Overview

### Simulation Flow

```
Configuration → System Building → Workload Execution → Result Collection
      ↓                ↓                  ↓                    ↓
  Load params    Create chiplets    Compute + Comms      Save state
  Input files    Build NoI          Schedule weights     Raw metrics
```

### Key Subsystems

#### Global Manager
The main orchestrator that coordinates all simulation components:
- Loads configuration and input files
- Initializes system components
- Manages simulation timeline
- Coordinates compute and communication simulators

#### Compute Simulator
Models analog in-memory compute operations:
- Layer computation latency
- Chiplet-specific compute characteristics
- Batch processing
- **Requirement:** IMC chiplet simulation calls the REST interface defined in `integrations/CIMLoop_API.py` (default `http://localhost:5000`). Ensure the modified CIMLoop Docker container with the API is running; otherwise IMC compute phases will fail. CMOS chiplets use the analytical backend and continue to run without CIMLoop.

#### Communication Simulator
Simulates Network-on-Interposer traffic:
- Integrates gem5 Garnet
- Cycle-accurate network simulation
- Packet routing and congestion modeling

#### System Model
Represents the chiplet-based architecture:
- Chiplet specifications and capabilities
- NoI topology and connectivity
- Resource allocation and management
  - Capacity tracking:
    - IMC chiplets: capacity units are crossbars; available memory = available_crossbars × memory_per_crossbar
    - CMOS chiplets: capacity units are weight units; `total_memory_weights` decrements as weights are mapped

#### Model Mapper
Maps DNN layers to chiplets:
- Partitioning algorithm
- Load balancing
- Communication minimization
  - Uses a unified capacity model:
    - IMC: layer requirement computed as crossbars needed (rows/cols/bits)
    - CMOS: layer requirement computed as total weights; allocation subtracts weight units from available capacity

#### Metric Computer
Computes performance metrics after the simulation is complete:
- Latency and throughput
- Network utilization
- Chiplet utilization
- Energy and power estimates

## Directory Structure

```
chiplet_simulator/
├── simulate.py                    # Unified CLI (simulate, reprocess, cross-analysis)
├── configs/                       # Configuration files
│   ├── experiments/               # Simulation configs (*.yaml)
│   └── cross_analysis/            # Cross-simulation analysis configs
├── assets/                        # Input files
│   ├── workloads/                 # Workload CSVs
│   ├── NoI_topologies/            # Network adjacency matrices
│   ├── chiplet_specs/             # Chiplet mappings and params
│   └── DNN_models/                # Model definitions
├── src/                           # Core simulator source
│   ├── run/                       # Orchestration
│   │   └── simulation_runner.py   # Entry from simulate.py
│   ├── sim/                       # Simulation engines
│   │   ├── compute_simulation/    # Compute modeling backends
│   │   └── communication_simulator.py
│   ├── core/                      # System modeling
│   ├── mapping/                   # DNN mapping algorithms
│   ├── post/                      # Post-processing and metrics
│   └── utils/                     # Utilities
├── helpers/                       # Helper scripts
├── integrations/                  # External tool integrations (gem5, etc.)
├── _results/                      # Simulation outputs (generated)
└── cache/                         # Cached results (generated)
```

## Related Documentation

- [Advanced Features](advanced-features.md) - Communication caching, weight strategies, pipelining
- [GEM5 Integration](gem5-integration.md) - Network simulator setup and usage
- [Helper Scripts](helper-scripts.md) - Utilities for workload and topology generation
