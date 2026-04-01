# Helper Scripts

Utility scripts for generating input files and managing model definitions.

All helper scripts are located in the [`helpers/`](../helpers/) directory.

## Workload Generation

### helpers_GenerateWorkload.py

Generate workload CSV files for simulation input.

**Usage:**
```bash
python3 helpers/helpers_GenerateWorkload.py
```

**Interactive prompts:**
- Number of inference requests
- DNN models to include
- Injection time distribution (uniform, poisson, burst, etc.)
- Batch sizes
- Output filename

**Example output** (`assets/workloads/custom_workload.csv`):
```csv
net_idx,inject_time_us,network,num_inputs
1,0,alexnet,1
2,150,resnet50,2
3,300,alexnet,1
4,450,vgg16,1
```

**Use cases:**
- Create synthetic workloads for benchmarking
- Generate stress tests with high injection rates
- Model realistic inference server traffic patterns

**Supported distributions:**
- **Uniform**: Evenly spaced injections
- **Poisson**: Random arrivals with specified rate
- **Burst**: Clusters of requests
- **Fixed**: Specific injection times

## Topology Generation

### helpers_GenerateAdjMatrix.py

Generate Network-on-Interposer topology adjacency matrices.

**Usage:**
```bash
python3 helpers/helpers_GenerateAdjMatrix.py
```

**Interactive prompts:**
- Topology type (mesh, torus, custom)
- Grid dimensions (rows x columns)
- Output filename

**Example output** (`assets/NoI_topologies/adj_matrix_8x8_mesh.csv`):
```csv
0,1,0,0,0,1,0,0,...
1,0,1,0,0,0,1,0,...
...
```

**Supported topologies:**
- **2D Mesh**: Grid with edge connections only
- **Torus**: Mesh with wraparound edges
- **Custom**: Specify connectivity pattern

**Matrix format:**
- Row i, Column j = 1 indicates link from chiplet i to chiplet j
- Diagonal typically 0 (no self-loops)
- Can be directed or undirected

**Use cases:**
- Quick generation of standard mesh topologies
- Explore different sizes (4x4, 8x8, 10x10, etc.)
- Baseline for custom topology modifications

## Model Definition Management


### create_def_from_cimloop.py

Import model definitions from CiMLoop format.

**Usage:**
```bash
python3 helpers/create_def_from_cimloop.py --input <cimloop_file> --output <model_name>
```

**Arguments:**
- `--input`: Path to CiMLoop model specification
- `--output`: Name for new model in simulator

**What it does:**
- Parses CiMLoop model format
- Converts to simulator's model definition format
- Handles layer mapping and parameter translation
- Validates converted model

**Use cases:**
- Import models from CiMLoop ecosystem
- Leverage existing model specifications
- Cross-tool compatibility

**Note:** CiMLoop is a cycle-accurate simulator for compute-in-memory architectures.

### create_new_model_def.py

Create a new model definition from scratch using guided prompts.

**Usage:**
```bash
python3 helpers/create_new_model_def.py
```

**What it does:**
- Scaffolds a model entry in `assets/DNN_models/model_definitions.py`
- Validates structure and basic parameters
- Prints next steps to integrate the model in workloads

## Chiplet Mapping Helpers

While there's no dedicated helper script for chiplet mappings, you can manually create them as YAML files in `assets/chiplet_specs/`.

**Example mapping file structure:**
```yaml
# mapping_64_uniform.yaml
1: SharedADC
2: SharedADC
3: SharedADC
# ... all chiplets with same type

# mapping_100_with_io.yaml
1: IO           # Dedicated I/O chiplets
2: IO
3: Accumulator  # Compute chiplets
4: SharedADC
# ... mixed types
```

**Available chiplet types** (defined in `assets/chiplet_specs/chiplet_params.py`):
Examples include `IMC_A`, `IMC_B`, `IMC_C`, `IMC_D`, `IMC_E`, `IMC_F`, `IO`, and `CMOS_Compute`.

## Related Documentation

- [Configuration Reference](../README.md#configuration) - Using generated files
- [Input Files](../README.md#input-files) - File format specifications
- [Components](components.md) - How helpers integrate with simulator