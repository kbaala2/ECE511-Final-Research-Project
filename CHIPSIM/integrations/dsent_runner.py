import os
import sys
import subprocess
import re
import json
import argparse
from configparser import ConfigParser, NoOptionError

DSENT_COMPILED = False

class _CaptureCOutput:
    def __init__(self):
        self.captured_output = ""

    def __enter__(self):
        self.original_stdout_fd = sys.stdout.fileno()
        self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        self.r, self.w = os.pipe()
        os.dup2(self.w, self.original_stdout_fd)
        os.close(self.w)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
        os.close(self.saved_stdout_fd)
        captured_output_bytes = os.read(self.r, 100000)
        os.close(self.r)
        self.captured_output = captured_output_bytes.decode('utf-8')

def _compile_dsent(gem5_dir):
    global DSENT_COMPILED
    if DSENT_COMPILED:
        return

    print("Compiling DSENT for first-time use...")
    # Use the provided gem5_dir explicitly (do not assume current working directory)
    dsent_build_dir = os.path.join(gem5_dir, "build/ext/dsent")
    dsent_src_dir_abs = os.path.join(gem5_dir, "ext/dsent")

    original_cwd = os.getcwd()
    try:
        if not os.path.exists(dsent_build_dir):
            os.makedirs(dsent_build_dir)
        
        os.chdir(dsent_build_dir)
        
        if not os.path.exists("Makefile"):
            cmake_cmd = ["cmake", dsent_src_dir_abs]
            process = subprocess.run(cmake_cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise RuntimeError(f"DSENT cmake failed: {process.stderr}")

        make_cmd = ["make", "-j" , str(os.cpu_count())]
        process = subprocess.run(make_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"DSENT make failed: {process.stderr}")

        print("DSENT compiled successfully.")
        DSENT_COMPILED = True
        
        sys.path.append(dsent_build_dir)

    finally:
        os.chdir(original_cwd)

def _parse_config(config_file):
    config = ConfigParser()
    if not config.read(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if not config.has_section("system.ruby.network"):
        raise ValueError("Ruby network not found in config file")

    if config.get("system.ruby.network", "type") != "GarnetNetwork":
        raise ValueError("Garnet network not used in config file")
    
    routers = config.get("system.ruby.network", "routers").split()
    int_links = config.get("system.ruby.network", "int_links").split()
    ext_links = config.get("system.ruby.network", "ext_links").split()

    network_config = {
        'num_vnet': config.getint("system.ruby.network", "number_of_virtual_networks"),
        'flit_size_bits': 8 * config.getint("system.ruby.network", "ni_flit_size"),
        'buffers_per_data_vc': config.getint("system.ruby.network", "buffers_per_data_vc"),
        'buffers_per_ctrl_vc': config.getint("system.ruby.network", "buffers_per_ctrl_vc")
    }
    return (config, network_config, routers, int_links, ext_links)

def _get_clock(obj, config, clock_override=None):
    if clock_override is not None:
        return clock_override
    try:
        obj_type = config.get(obj, "type")
        if obj_type == "SrcClockDomain":
            return config.getint(obj, "clock")
        if obj_type == "DerivedClockDomain":
            source = config.get(obj, "clk_domain")
            divider = config.getint(obj, "clk_divider")
            return _get_clock(source, config) / divider
    except NoOptionError:
        pass

    try:
        source = config.get(obj, "clk_domain")
        return _get_clock(source, config)
    except NoOptionError:
        if obj != "system.ruby.network":
            return _get_clock("system.ruby.network", config)
        else:
            raise ValueError(f"Could not determine clock for {obj}")

def _get_router_config(config, router, int_links, ext_links, clock_override=None):
    router_id = int(router.partition("routers")[2])
    num_ports = sum(1 for link in int_links if config.get(link, "src_node") == router or config.get(link, "dst_node") == router)
    num_ports += sum(1 for link in ext_links if config.get(link, "int_node") == router)

    clock_period = _get_clock(router, config, clock_override)
    frequency = 1e12 / float(clock_period)

    return {
        'router_id': router_id,
        'num_inports': num_ports,
        'num_outports': num_ports,
        'vcs_per_vnet': config.getint(router, "vcs_per_vnet"),
        'frequency': int(frequency)
    }

def _get_link_config(config, link, clock_override=None):
    clock_period = _get_clock(link, config, clock_override)
    frequency = 1e12 / float(clock_period)
    return {
        'frequency': int(frequency),
        'length': 4e-3,
        'delay': float(1 / frequency)
    }

def _parse_network_stats(stats_file):
    stats = {}
    with open(stats_file, 'r') as f:
        for line in f:
            if re.match("simTicks", line):
                stats['simTicks'] = int(re.split(r'\s+', line)[1])
            elif re.match("simSeconds", line):
                stats['simSeconds'] = float(re.split(r'\s+', line)[1])
            elif re.match("simFreq", line):
                stats['simFreq'] = float(re.split(r'\s+', line)[1])
    return stats

def _parse_router_stats(stats_file, router):
    stats = {'buffer_writes': 0, 'buffer_reads': 0, 'crossbar_activity': 0, 'sw_in_arb_activity': 0, 'sw_out_arb_activity': 0}
    with open(stats_file, 'r') as f:
        for line in f:
            if re.match(router, line):
                if re.search("buffer_writes", line):
                    stats['buffer_writes'] = int(re.split(r'\s+', line)[1])
                elif re.search("buffer_reads", line):
                    stats['buffer_reads'] = int(re.split(r'\s+', line)[1])
                elif re.search("crossbar_activity", line):
                    stats['crossbar_activity'] = int(re.split(r'\s+', line)[1])
                elif re.search("sw_input_arbiter_activity", line):
                    stats['sw_in_arb_activity'] = int(re.split(r'\s+', line)[1])
                elif re.search("sw_output_arbiter_activity", line):
                    stats['sw_out_arb_activity'] = int(re.split(r'\s+', line)[1])
    return stats

def _parse_link_stats(stats_file, sim_ticks):
    stats = {}
    with open(stats_file, 'r') as f:
        for line in f:
            if re.search("avg_link_utilization", line):
                stats['activity'] = int(float(re.split(r'\s+', line)[1]) * sim_ticks)
    return stats

def _update_router_config_stats(dsent, network_config, router_config, network_stats, router_stats):
    num_cycles = int(network_stats['simTicks'] / 1000)
    dsent.updateRouterConfigStats(
        network_config['tech_model'], router_config['frequency'], network_config['flit_size_bits'],
        router_config['num_inports'], router_config['num_outports'], network_config['num_vnet'],
        router_config['vcs_per_vnet'], network_config['buffers_per_ctrl_vc'],
        network_config['buffers_per_data_vc'], num_cycles, router_stats['buffer_writes'],
        router_stats['buffer_reads'], router_stats['sw_in_arb_activity'],
        router_stats['sw_out_arb_activity'], router_stats['crossbar_activity']
    )

def _update_link_config_stats(dsent, network_config, link_config, network_stats, link_stats):
    num_cycles = int(network_stats['simTicks'] / 1000)
    dsent.updateLinkConfigStats(
        network_config['tech_model'], link_config['frequency'], network_config['flit_size_bits'],
        link_config['length'], link_config['delay'], num_cycles, link_stats.get('activity', 0)
    )

def _parse_dsent_output(output):
    power, area = 0.0, 0.0
    lines, total_power_idx, area_idx = output.split('\n'), -1, -1
    for i, line in enumerate(lines):
        if line.strip() == 'Total Power:': total_power_idx = i
        if line.strip() == 'Area:': area_idx = i
    if total_power_idx != -1:
        for i in range(total_power_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip() or not line.startswith('    '): break
            match = re.search(r'power \(W\):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if match: power += float(match.group(1))
    else:
        for line in lines:
            match = re.search(r'power \(W\):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if match: power += float(match.group(1))
    if area_idx != -1:
        for i in range(area_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip() or not line.startswith('    '): break
            match = re.search(r':\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if match: area += float(match.group(1))
    return power, area

def _parse_dsent_output_detailed(output):
    data, current_section = {}, None
    lines = output.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line: continue
        if line.endswith(':'):
            current_section = line[:-1]
            data[current_section] = {}
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
                sub_line = lines[i].strip()
                i += 1
                if not sub_line: continue
                key = sub_line.split(':')[0].strip()
                match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', sub_line)
                if match: data[current_section][key] = float(match.group(1))
    return data

def run_dsent_and_get_results(tech_node, gem5_dir):
    _compile_dsent(gem5_dir)
    import dsent

    sim_dir = os.path.join(gem5_dir, "m5out")
    config_file = os.path.join(sim_dir, "config.ini")
    stats_file = os.path.join(sim_dir, "stats.txt")
    router_config_file = os.path.join(gem5_dir, "ext/dsent/configs/garnet_router.cfg")
    link_config_file = os.path.join(gem5_dir, "ext/dsent/configs/garnet_link.cfg")

    if not os.path.isfile(config_file) or not os.path.isfile(stats_file):
        print(f"DSENT Error: config/stats file not found in {os.path.abspath(sim_dir)}")
        return {}

    all_config, network_config, routers, int_links, ext_links = _parse_config(config_file)
    
    tech_model_path = os.path.join(gem5_dir, "ext/dsent/tech/tech_models/")
    if tech_node in ["45", "32", "22"]:
        network_config['tech_model'] = tech_model_path + "Bulk" + tech_node + "LVT.model"
    elif tech_node == "11":
        network_config['tech_model'] = tech_model_path + "TG" + tech_node + "LVT.model"
    else:
        raise ValueError(f"Unknown DSENT Tech Model: {tech_node}")

    network_stats = _parse_network_stats(stats_file)
    dsent.initialize(router_config_file)
    router_results, router_outputs_text = [], []
    for router in routers:
        router_config = _get_router_config(all_config, router, int_links, ext_links)
        router_stats = _parse_router_stats(stats_file, router)
        _update_router_config_stats(dsent, network_config, router_config, network_stats, router_stats)
        with _CaptureCOutput() as capture:
            dsent.run()
        router_outputs_text.append(capture.captured_output)
        router_results.append({
            "router_id": router_config['router_id'],
            "results": _parse_dsent_output_detailed(capture.captured_output)
        })
    dsent.finalize()

    dsent.initialize(link_config_file)
    link_config = {}
    for link in int_links + ext_links:
        link_config = _get_link_config(all_config, link)
    
    sim_ticks = network_stats['simTicks']
    link_stats = _parse_link_stats(stats_file, sim_ticks)
    _update_link_config_stats(dsent, network_config, link_config, network_stats, link_stats)
    with _CaptureCOutput() as capture:
        dsent.run()
    link_results_text = capture.captured_output
    dsent.finalize()
    
    link_results = _parse_dsent_output_detailed(link_results_text)
    
    total_router_power, total_router_area = 0, 0
    for res_text in router_outputs_text:
        power, area = _parse_dsent_output(res_text)
        total_router_power += power
        total_router_area += area
    link_power, link_area = _parse_dsent_output(link_results_text)
    
    total_power = total_router_power + link_power
    total_area_m2 = total_router_area + link_area

    print("✅ DSENT simulation completed successfully.")
    return {
        "routers": router_results, "links": link_results,
        "totals": {
            "router_power_W": total_router_power, "router_area_mm2": total_router_area * 1e6,
            "link_power_W": link_power, "link_area_mm2": link_area * 1e6,
            "total_power_W": total_power, "total_area_mm2": total_area_m2 * 1e6
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Run DSENT simulation in a separate process.")
    parser.add_argument("--gem5-dir", required=True, help="Path to the gem5 directory.")
    parser.add_argument("--tech-node", required=True, help="DSENT technology node.")
    parser.add_argument("--output-file", required=True, help="Path to write the output JSON results.")
    args = parser.parse_args()

    original_dir = os.getcwd()
    try:
        # Run DSENT using explicit gem5 directory paths (no reliance on CWD)
        results = run_dsent_and_get_results(args.tech_node, args.gem5_dir)
        
        # Ensure we are in the original directory before writing file
        os.chdir(original_dir)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"DSENT runner finished. Results written to {os.path.abspath(args.output_file)}")

    except Exception as e:
        # Change back to original dir on error too
        os.chdir(original_dir)
        print(f"Error during DSENT runner execution: {e}", file=sys.stderr)
        # Write empty result to signal failure
        with open(args.output_file, 'w') as f:
            json.dump({}, f)
        sys.exit(1)

if __name__ == "__main__":
    main() 