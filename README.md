# AutoMap: LLM-Driven DNN Mapping Optimization for CHIPSIM

AutoMap is a multi-agent LLM pipeline that automatically generates, validates, and evaluates improved DNN-to-chiplet mapping algorithms for the [CHIPSIM](https://github.com/LukasPfromm/CHIPSIM) simulator. Instead of manually tuning mapping heuristics, AutoMap uses an LLM to analyze the existing nearest-neighbor mapping algorithm, generate drop-in replacement code, and iteratively refine it based on simulation feedback.

## How It Works

```
START → baseline → analyzer → optimizer → validator ──(invalid)──→ optimizer (retry)
            ↑                                │
            │                             (valid)
            │                                ↓
            │                           injector → runtime_handler
            │                                            │
            │                          ┌─────────────────┼─────────────────┐
            │                       (error)           (success)        (max retries)
            │                          ↓                 ↓                 ↓
            │                      optimizer         evaluator         evaluator
            │                                            │
            │                    ┌───────────────────────┼──────────────┐
            │                 (improve)             (no improve)    (max iters)
            └─────────────────────┘                     ↓                ↓
                                                    analyzer            END
```

1. **Baseline** — Establishes reference metrics by running CHIPSIM with the original nearest-neighbor mapper (or loading pre-computed results)
2. **Analyzer** — LLM identifies weaknesses in the current mapping algorithm for the given chiplet configuration
3. **Optimizer** — LLM generates a complete drop-in replacement mapping function
4. **Validator** — AST-based checks verify function signature, return types, and contract compliance
5. **Injector** — Deterministically wires the generated function into CHIPSIM's source code
6. **Runtime Handler** — Executes the simulator, catches crashes, feeds tracebacks back for retry
7. **Evaluator** — Parses simulation metrics, compares against baseline, decides whether to continue

The loop repeats for a configurable number of iterations, with the LLM receiving feedback from prior attempts to avoid repeating failed strategies.

## Prerequisites

- **Python 3.10+**
- **CHIPSIM** simulator cloned and functional ([setup instructions](https://github.com/LukasPfromm/CHIPSIM))
- **LLM access** via one of:
  - [OpenRouter](https://openrouter.ai/) API key (recommended — used for DeepSeek V3.2 / V4 Pro)
  - [Groq](https://console.groq.com/) API key (free tier — Llama 3.3 70B)
  - [Ollama](https://ollama.ai/) running locally (Qwen2.5-Coder 32B or similar)

## Installation

```bash
# Clone the repo (if not already)
git clone <your-repo-url>
cd ECE511-Final-Research-Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
langgraph>=0.4.0
langchain-core>=0.3.0
langchain-groq>=0.3.0
langchain-ollama>=0.3.0
langchain-openai>=0.3.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Choose one (or multiple) depending on your LLM provider:
OPENROUTER_API_KEY=sk-or-...      # For DeepSeek V3.2 / V4 Pro via OpenRouter
GROQ_API_KEY=gsk_...              # For Llama 3.3 70B via Groq
# Ollama requires no API key — just run `ollama serve`
```

## Usage

### Single Configuration

```bash
# Default: config_1, 3 iterations, 3 retries
python agent.py

# Specify a config
python agent.py --config resnet50_hetero_checkerboard

# More iterations
python agent.py --config resnet50_hetero_checkerboard --max-iterations 5

# All options
python agent.py \
  --config resnet50_hetero_checkerboard \
  --max-iterations 5 \
  --max-retries 3 \
  --max-runtime-retries 3
```

The `--config` argument is the filename (without `.yaml`) from `CHIPSIM/configs/experiments/`. If the config is not found, the agent lists all available configs and exits.

### Batch Run (Multiple Configurations)

```bash
chmod +x run_all_configs.sh
./run_all_configs.sh
```

This runs the agent sequentially for each configuration listed in the script, saving per-config logs to `run_logs/` and printing a consolidated summary at the end.

To run in the background:

```bash
nohup ./run_all_configs.sh > run_all_output.log 2>&1 &
tail -f run_all_output.log
```

### CLI Arguments Reference

| Argument | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config_1` | CHIPSIM config name (without `.yaml`) |
| `--max-iterations` | `-n` | `3` | Maximum optimization iterations |
| `--max-retries` | `-r` | `3` | Max validation retries per iteration |
| `--max-runtime-retries` | | `3` | Max runtime retries per iteration |

## Output Files

After a run, the following files are generated:

### Agent Outputs (in the working directory)

| File | Description |
|---|---|
| `last_generated_mapper.py` | The most recent LLM-generated mapping function, with metadata comments (config, function name, iteration, improvement percentages) |
| `optimization_history_<config>.yaml` | Full optimization history: baseline metrics, best metrics, and per-iteration entries with outcomes, metrics, errors, and improvement percentages |

### CHIPSIM Results (under `CHIPSIM/_results/`)

| Path | Description |
|---|---|
| `_results/raw_results/<config>_<function>/` | Raw simulation output for each iteration |
| `_results/formatted_results/<config>_<function>/` | Formatted results with plots and metrics |
| `_results/<config>_<function>_stdout.log` | Simulator stdout for each iteration |
| `_results/<config>_<function>_stderr.log` | Simulator stderr for each iteration |
| `_results/raw_results/<config>_baseline/` | Baseline simulation raw output |
| `_results/formatted_results/<config>_baseline/` | Baseline formatted results |

The key metrics file is at:
```
_results/formatted_results/<config>_<function>/formatted_comparison_metrics/Approach_Comparison_Metrics.txt
```

### Batch Run Logs

| Path | Description |
|---|---|
| `run_logs/<config>.log` | Full agent output for each config in a batch run |

## Configuration Guide

### Changing the LLM Provider

The LLM is instantiated once in `agent.py`. To switch providers, modify the `llm = ...` line:

```python
# OpenRouter (DeepSeek V3.2 / V4 Pro) — recommended
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat-v3-0324",  # or "deepseek/deepseek-v4-pro"
)

# Groq (Llama 3.3 70B) — free tier
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Ollama (local) — no API key needed
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5-coder:32b", num_ctx=8192)
```

No other code changes are needed — all nodes use the same `llm` object.

### Changing CHIPSIM Paths

Update the constants at the top of `agent.py`:

```python
CHIPSIM_ROOT = "/path/to/your/CHIPSIM"
MAPPER_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper.py"
MAPPER_ORIGINAL_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper_original.py"
PARTITIONER_PATH = f"{CHIPSIM_ROOT}/src/mapping/layer_partitioner.py"
```

### Pre-Computed Baselines

If baselines have already been run and results stored externally (e.g., in a shared `performance_results/` directory), you can modify `run_baseline()` to read from that path instead of re-running the simulation. Change the baseline metrics path to point to your pre-computed results:

```python
baseline_metrics_path = os.path.join(
    "/path/to/performance_results/baseline/heterogenous",
    config_name,
    "formatted_comparison_metrics",
    "Approach_Comparison_Metrics.txt"
)
```

### Modifying the Mapper Contract

The `MAPPER_CONTRACT` string in `agent.py` defines the rules the LLM must follow when generating code. You can modify it to:

- Add or remove allowed data accessors (e.g., if new chiplet properties are available)
- Change the required function signature (if CHIPSIM's mapper interface changes)
- Add constraints (e.g., "do not use more than N chiplets")
- Relax constraints for experimentation

### Modifying the Evaluation Criteria

The evaluator in `evaluate_results()` uses `total_latency` as the primary metric for deciding improvements. To change this (e.g., to optimize for communication latency or energy), modify the `is_improvement` check:

```python
# Default: optimize for total latency
is_improvement = latest_metrics["total_latency"] < best["total_latency"]

# Alternative: optimize for communication latency
is_improvement = latest_metrics["comm_latency"] < best["comm_latency"]
```

### Adding New CHIPSIM Configurations

1. Create a new YAML config in `CHIPSIM/configs/experiments/`
2. Ensure the referenced input files exist (workload CSV, adjacency matrix, chiplet mapping YAML)
3. Run: `python agent.py --config your_new_config`

## Project Structure

```
ECE511-Final-Research-Project/
├── agent.py                          # Main agent pipeline
├── run_all_configs.sh                # Batch execution script
├── requirements.txt                  # Python dependencies
├── .env                              # API keys (git-ignored)
├── last_generated_mapper.py          # Most recent generated function
├── optimization_history_*.yaml       # Per-config optimization logs
├── run_logs/                         # Batch run logs
│
├── CHIPSIM/                          # CHIPSIM simulator
│   ├── configs/experiments/          # Simulation config YAMLs
│   ├── src/mapping/
│   │   ├── model_mapper.py           # Active mapper (modified by agent)
│   │   ├── model_mapper_original.py  # Pristine backup (never modified)
│   │   └── layer_partitioner.py      # Layer partitioning logic
│   ├── assets/
│   │   ├── chiplet_specs/            # Chiplet mapping YAMLs
│   │   ├── DNN_models/               # Model definitions
│   │   ├── NoI_topologies/           # Adjacency matrices
│   │   └── workloads/                # Workload CSVs
│   └── _results/
│       ├── raw_results/              # Per-run raw simulation output
│       └── formatted_results/        # Per-run formatted metrics & plots
│
└── performance_results/              # Pre-computed baseline results
    └── baseline/
        └── heterogenous/
            └── <config_name>/
                └── formatted_comparison_metrics/
                    └── Approach_Comparison_Metrics.txt
```

## Troubleshooting

**"ERROR: Config file not found"** — The config name doesn't match any YAML in `CHIPSIM/configs/experiments/`. Check the listed available configs and use the name without the `.yaml` extension.

**Validation keeps failing** — The LLM is producing code that doesn't match the contract. Try a more capable model (DeepSeek V4 Pro > V3.2 > Llama 3.3 70B > local models). Check `last_generated_mapper.py` to see what the LLM is producing.

**Runtime errors in `global_manager.py`** — This is a CHIPSIM initialization error, not a mapper error. Verify the config's chiplet mapping, adjacency matrix, and workload files are correct. If using IMC chiplets, ensure the CIMLoop Docker container is running.

**Stale results / metrics from wrong run** — Delete `CHIPSIM/_results/` and re-run. The evaluator reads from a path based on the config and function name; old results from prior runs with different configs can cause mismatches.

**Slow simulation** — If using `capture_output=True` in subprocess calls, switch to file-based stdout/stderr logging. Also ensure `PYTHONUNBUFFERED=1` is set in the simulation environment.

**Rate limited on Groq** — Add `time.sleep(3)` after each `llm.invoke()` call, or switch to OpenRouter which has higher limits.

**Mapper backup is corrupted** — Delete `model_mapper_original.py` and restore from git:
```bash
rm CHIPSIM/src/mapping/model_mapper_original.py
cd CHIPSIM && git checkout -- src/mapping/model_mapper.py
```

