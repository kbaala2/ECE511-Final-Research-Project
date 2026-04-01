# Advanced Features

This document covers advanced simulation features and configuration options.

## Communication Caching

Enable communication result caching to speed up repeated simulations with identical traffic patterns.

**Configuration:**
```yaml
simulation:
  core_settings:
    enable_comm_cache: true
```

**How it works:**
- Network simulation results are cached based on traffic pattern hash
- Subsequent simulations with the same pattern reuse cached results
- Dramatically reduces simulation time for parameter sweeps
- Cache stored in `cache/` directory

**When to use:**
- Parameter exploration with fixed workloads
- Iterative design space exploration
- Debugging compute logic without re-simulating network

**Limitations:**
- Only caches identical traffic patterns
- Cache invalidated if topology or network parameters change
- Requires sufficient disk space for large experiments

## Weight Loading Strategies

Configure how model weights are transferred to chiplets.

### All At Once
```yaml
simulation:
  core_settings:
    weight_loading_strategy: "all_at_once"
```

- Loads all model weights at simulation start
- One-time upfront communication cost
- Best for repeated inference of same model
- Higher initial latency, lower per-inference cost

### Just In Time
```yaml
simulation:
  core_settings:
    weight_loading_strategy: "just_in_time"
```

- Loads weights immediately before layer execution
- Distributed communication throughout execution
- Best for mixed workloads or single inference
- Lower initial latency, higher per-inference cost

### Weight Stationary Mode
```yaml
simulation:
  core_settings:
    weight_stationary: true
```

- Weights remain on chiplets between inferences
- Only loads weights once per model (not per inference)

```yaml
simulation:
  core_settings:
    weight_stationary: false
```

- Weights reloaded for each inference
- Higher communication overhead

## Communication Methods

### Pipelined Communication
```yaml
simulation:
  core_settings:
    comm_method: "pipelined"
```

**Behavior:**
- Overlaps execution of model layers by independent inputs
- Layer N of input A can execute while layer N+1 of input B executes
- Maximizes throughput
- More complex scheduling

### Non-Pipelined Communication
```yaml
simulation:
  core_settings:
    comm_method: "non-pipelined"
```

## CMOS Analytical Compute

CMOS chiplets use an analytical compute backend (no crossbars). Compute latency and energy are derived from total MACs and per-chiplet parameters.

- `CMOS_Compute` parameters (see `assets/chiplet_specs/chiplet_params.py`):
  - `macs_per_second`, `energy_per_mac`, and a hard-coded `total_memory_weights`
- During mapping, weights assigned to a CMOS chiplet decrement its available weight capacity.

## Capacity Tracking (IMC vs CMOS)

The mapper uses a unified capacity model:
- **IMC**: capacity units are crossbars. Available memory derives from available crossbars × memory per crossbar. Layer requirements are calculated as crossbars needed.
- **CMOS**: capacity units are weight units. Available capacity equals remaining `total_memory_weights`. Layer requirements are the number of weights.

This enables consistent “ever fit” (total capacity) and “fit now” (available capacity) checks across chiplet types.

**Behavior:**
- Sequential compute-then-communicate for each input
- Layer completes fully (compute + communication) before next layer starts

## DSENT Integration

Enable power and area estimation for the Network-on-Interposer using DSENT.

```yaml
simulation:
  core_settings:
    enable_dsent: true
  dsent_parameters:
    dsent_tech_node: "32"  # Technology node in nm
```

**Requirements:**
- gem5 built with DSENT integration
- DSENT configuration files in `gem5/ext/dsent/configs/`

**Output:**
- Per-router power consumption
- Link power consumption
- Total NoI power and area estimates

**Performance note:** Enabling DSENT increases simulation time.

See [GEM5 Integration](gem5-integration.md) for DSENT setup details.

## Warmup Period

Exclude initial warmup period from metric collection to get steady-state performance.

```yaml
# During simulation
simulation:
  core_settings:
    warmup_period_us: 1000.0  # First 1000 microseconds excluded

# During post-processing only (if reprocessing)
post_processing:
  warmup_period_us: 1000.0
```

## Cooldown Period

Exclude the last segment of the run from metric collection (e.g., to avoid tail effects).

```yaml
post_processing:
  cooldown_period_us: 0.0  # Exclude last X microseconds (0.0 disables)
```

**Use cases:**
- Eliminate cold-start effects
- Measure steady-state throughput

**Note:** Post-processor can also filter warmup period. See configuration documentation.

## Network Operation Frequency

Configure the NoI operating frequency:

```yaml
simulation:
  hardware_parameters:
    network_operation_frequency_hz: 1000000000  # 1 GHz
```

Affects:
- Communication latency
- Power consumption (if DSENT enabled)
- Throughput calculations

## Packet and Activation Sizing

Configure data representation and packet sizes:

```yaml
simulation:
  hardware_parameters:
    bits_per_activation: 8    # Activation data width
    bits_per_packet: 128      # Network packet size
```

**bits_per_activation:**
- Affects communication volume
- Typical values: 8, 16, 32 bits

**bits_per_packet:**
- Affects packet count and network granularity

## Advanced GEM5 Parameters

Fine-tune gem5 Garnet simulation:

```yaml
simulation:
  gem5_parameters:
    gem5_sim_cycles: 500000000    # Maximum simulation cycles
    gem5_injection_rate: 0.0      # For synthetic traffic (use 0.0 for trace)
    gem5_ticks_per_cycle: 1000    # Time resolution
```

**gem5_sim_cycles:**
- Can be ignored. Simulations will run until complete. 

**gem5_ticks_per_cycle:**
- gem5 internal time resolution
- Do not modify! Simulator assumes 1000 ticks per cycle. 

## Related Documentation

- [Configuration Reference](../README.md#configuration) - Full parameter list
- [GEM5 Integration](gem5-integration.md) - Network simulator details
- [Post-Processing](../README.md#post-processing) - Analysis features
- [Components](components.md) - Core simulator architecture
