# Copyright (c) 2014 Mark D. Hill and David A. Wood
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import string
import subprocess
import sys
from configparser import ConfigParser, NoOptionError
import re
import argparse

# Compile DSENT to generate the Python module and then import it.
# This script assumes it is executed from the gem5 root.
print("Attempting compilation")
from subprocess import call

# Determine paths relative to the script's location, assuming it's in gem5/util
_script_abs_path = os.path.realpath(__file__)
_script_dir = os.path.dirname(_script_abs_path)
# Assuming gem5 root is one level above the script's directory
_gem5_root_inferred = os.path.abspath(os.path.join(_script_dir, ".."))

# This is the name of the dsent source directory, relative to the gem5 root.
# It's used to construct the argument for cmake when cmake is run from the build directory.
# e.g., cmake ../../../<dsent_source_subdir_name_rel_to_gem5_root>
dsent_source_subdir_name_rel_to_gem5_root = "ext/dsent"

# This is the target build directory for dsent.
# The script will chdir into this directory to run cmake and make.
# It will be created if it doesn't exist.
_dsent_build_path_rel_to_gem5_root = "build/ext/dsent"
# Absolute path to the dsent build directory
dsent_build_dir_abs = os.path.join(_gem5_root_inferred, _dsent_build_path_rel_to_gem5_root)


if not os.path.exists(dsent_build_dir_abs):
    os.makedirs(dsent_build_dir_abs)

# Save the current working directory before changing it, to restore if necessary,
# though the script later changes CWD to gem5_root_inferred.
_original_cwd = os.getcwd()
os.chdir(dsent_build_dir_abs)

# The cmake command needs the path to the dsent sources relative to the dsent_build_dir_abs.
# If dsent_build_dir_abs is GEM5_ROOT/build/ext/dsent, then ../../../ gets to GEM5_ROOT.
# Appending dsent_source_subdir_name_rel_to_gem5_root ("ext/dsent") gives GEM5_ROOT/ext/dsent.
error = call(["cmake", "../../../" + dsent_source_subdir_name_rel_to_gem5_root])
if error:
    print("Failed to run cmake")
    # Restore CWD before exiting if there was an error during compilation setup
    os.chdir(_original_cwd)
    exit(-1)

error = call(["make"])
if error:
    print("Failed to run make")
    # Restore CWD before exiting
    os.chdir(_original_cwd)
    exit(-1)

print("Compiled dsent")
# Change CWD to the inferred gem5 root directory.
# From dsent_build_dir_abs (GEM5_ROOT/build/ext/dsent), ../../../ leads to GEM5_ROOT.
os.chdir("../../../") # CWD is now _gem5_root_inferred
# sys.path.append needs the path to the dsent Python module,
# which is typically in build/ext/dsent relative to the gem5 root.
sys.path.append(os.path.join("build", "ext", "dsent")) # Appends GEM5_ROOT/build/ext/dsent

import dsent

# Context manager to capture C-level stdout, since dsent is a C++ extension
# that writes directly to file descriptors.
class CaptureCOutput:
    def __init__(self):
        self.captured_output = ""

    def __enter__(self):
        # The original file descriptor for stdout
        self.original_stdout_fd = sys.stdout.fileno()
        # Save a copy of the original stdout file descriptor
        self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        # Create a pipe
        self.r, self.w = os.pipe()
        # Duplicate the write-end of the pipe to stdout file descriptor
        os.dup2(self.w, self.original_stdout_fd)
        # Close the original write-end of the pipe
        os.close(self.w)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush the C-level buffer
        sys.stdout.flush()
        # Restore the original stdout file descriptor
        os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
        # Close the saved copy of the original stdout file descriptor
        os.close(self.saved_stdout_fd)
        # Read the captured output from the read-end of the pipe
        captured_output_bytes = os.read(self.r, 100000)
        os.close(self.r)
        self.captured_output = captured_output_bytes.decode('utf-8')


# Parse gem5 config.ini file for the configuration parameters related to
# the on-chip network.
def parseConfig(config_file):
    config = ConfigParser()
    if not config.read(config_file):
        print(("ERROR: config file '", config_file, "' not found"))
        sys.exit(1)

    if not config.has_section("system.ruby.network"):
        print(("ERROR: Ruby network not found in '", config_file))
        sys.exit(1)

    if config.get("system.ruby.network", "type") != "GarnetNetwork":
        print(("ERROR: Garnet network not used in '", config_file))
        sys.exit(1)

    routers = config.get("system.ruby.network", "routers").split()
    int_links = config.get("system.ruby.network", "int_links").split()
    ext_links = config.get("system.ruby.network", "ext_links").split()

    network_config = {}
 
    network_config['num_vnet'] = \
        config.getint("system.ruby.network", "number_of_virtual_networks")

    network_config['flit_size_bits'] = \
        8 * config.getint("system.ruby.network", "ni_flit_size")

    # TODO: this should be a per router parameter:
    network_config['buffers_per_data_vc'] = \
        config.getint("system.ruby.network", "buffers_per_data_vc")

    network_config['buffers_per_ctrl_vc'] = \
        config.getint("system.ruby.network", "buffers_per_ctrl_vc")

    ## Update technology node
    tech_model_path = "ext/dsent/tech/tech_models/"
    # tech = sys.argv[5] # Old way of getting tech
    # Use tech from parsed args later in the script, or pass it to this function
    # For now, we'll adjust this when we use parsed_args.tech_node

    # if (tech == "45" or tech == "32" or tech == "22"):
    #     network_config['tech_model'] = \
    #         tech_model_path + "Bulk" + tech + "LVT.model"
    # elif (tech == "11"):
    #     network_config['tech_model'] = \
    #         tech_model_path + "TG" + tech + "LVT.model"
    # else:
    #     print("Unknown Tech Model. Supported models: 45, 32, 22, 11")
    #     exit(-1)

    return (config, network_config, routers, int_links, ext_links)

def getRouterConfig(config, router, int_links, ext_links, clock_override=None):

    router_id = int(router.partition("routers")[2])
    num_ports = 0

    for int_link in int_links:
        if config.get(int_link, "src_node") == router or \
           config.get(int_link, "dst_node") == router:
           num_ports += 1

    for ext_link in ext_links:
        if config.get(ext_link, "int_node") == router:
           num_ports += 1

    router_config = {}
    router_config['router_id']      = router_id
    router_config['num_inports']    = num_ports
    router_config['num_outports']   = num_ports

    router_config['vcs_per_vnet'] = \
        config.getint(router, "vcs_per_vnet")

    # add buffers_per_ctrl_vc and buffers_per_data_vc her

    # FIXME: Clock period units are ns in tester, and ps in full-system
    # Make it consistent in gem5
    clock_period = getClock(router, config, clock_override)
    frequency = 1e12 / float(clock_period)
    router_config['frequency'] = int(frequency)

    return router_config

def getLinkConfig(config, link, clock_override=None):

    link_config = {}

    # Frequency (Hz)
    clock_period = getClock(link, config, clock_override)
    frequency = 1e12 / float(clock_period)
    link_config['frequency'] = int(frequency)

    # Length (m)
    # FIXME: will be part of topology file and appear in config.ini
    length = 4e-3
    link_config['length'] = float(length)

    # Delay (s)
    # Delay of the wire need not be 1.0 / Frequency
    # wire could run faster
    link_config['delay'] = float(1 / frequency)

    return link_config

def parseNetworkStats(stats_file):

    try:
        lines = open(stats_file, 'r')
    except IOError:
        print("Failed to open ", stats_file, " for reading")
        exit(-1)

    network_stats = {}

    for line in lines:
        if re.match("simSeconds", line):
            network_stats['simSeconds'] = float(re.split('\s+', line)[1])
        if re.match("simTicks", line):
            network_stats['simTicks'] = int(re.split('\s+', line)[1])
        if re.match("simFreq", line):
            network_stats['simFreq'] = float(re.split('\s+', line)[1])

    lines.close()
    return network_stats

def parseRouterStats(stats_file, router):

    try:
        lines = open(stats_file, 'r')
    except IOError:
        print("Failed to open ", stats_file, " for reading")
        exit(-1)

    router_stats = {
        'buffer_writes': 0,
        'buffer_reads': 0,
        'crossbar_activity': 0,
        'sw_in_arb_activity': 0,
        'sw_out_arb_activity': 0
    }
    for line in lines:
        if re.match(router, line):
            if re.search("buffer_writes", line):
                router_stats['buffer_writes'] = int(re.split('\s+', line)[1])
            if re.search("buffer_reads", line):
                router_stats['buffer_reads'] = int(re.split('\s+', line)[1])
            if re.search("crossbar_activity", line):
                router_stats['crossbar_activity'] = int(re.split('\s+', line)[1])
            if re.search("sw_input_arbiter_activity", line):
                router_stats['sw_in_arb_activity'] = int(re.split('\s+', line)[1])
            if re.search("sw_output_arbiter_activity", line):
                router_stats['sw_out_arb_activity'] = int(re.split('\s+', line)[1])

    return router_stats

def parseLinkStats(stats_file, sim_ticks):

    try:
        lines = open(stats_file, 'r')
    except IOError:
        print("Failed to open ", stats_file, " for reading")
        exit(-1)

    link_stats = {}
    for line in lines:
        if re.search("avg_link_utilization", line):
            link_stats['activity'] = \
                int(float(re.split('\s+', line)[1]) * sim_ticks)

    return link_stats

def getClock(obj, config, clock_override=None):
    # Use clock period specified from command line
    # if available
    # if len(sys.argv) > 6: # Old way
    #     clock = sys.argv[6]
    #     return clock
    if clock_override is not None: # New way using parsed args
        return clock_override

    try:
        # Check if the object itself is a clock domain
        obj_type = config.get(obj, "type")
        if obj_type == "SrcClockDomain":
            return config.getint(obj, "clock")
        if obj_type == "DerivedClockDomain":
            source = config.get(obj, "clk_domain") # This clk_domain should exist for DerivedClockDomain
            divider = config.getint(obj, "clk_divider")
            return getClock(source, config) / divider
        # If obj_type is something else, it's not a direct clock source.
        # Proceed to get its clk_domain attribute in the next try block.
    except NoOptionError:
        # This means 'obj' section might have been found, but it doesn't have a 'type' attribute,
        # or 'type' is not Src/Derived, or (less likely here) 'clk_domain'/'clk_divider' is missing for a Derived type.
        # We'll proceed to the general clk_domain lookup.
        pass

    try:
        # General case: object points to a clock domain via 'clk_domain' attribute
        source = config.get(obj, "clk_domain")
        return getClock(source, config)
    except NoOptionError:
        # Fallback: 'obj' section was found but has no 'clk_domain' option.
        # This is where the error for 'system.ruby.network.int_links000' occurs.
        if obj != "system.ruby.network":
            try:
                # Assuming 'system.ruby.network' will have a resolvable clock
                return getClock("system.ruby.network", config)
            except Exception as e:
                print(f"ERROR: Fallback to 'system.ruby.network' clock also failed for '{obj}'. Original object '{obj}' missing 'clk_domain'. Error: {e}")
                # Re-raise the original NoOptionError to indicate clk_domain was missing for 'obj'
                raise NoOptionError("clk_domain", obj) from e
        else:
            # If 'obj' IS 'system.ruby.network' and it also has no clk_domain (and wasn't a Src/Derived type),
            # then we cannot determine the clock.
            print(f"ERROR: Critical component '{obj}' has no 'clk_domain' and is not a recognized clock type.")
            raise # Re-raise the NoOptionError for clk_domain on 'system.ruby.network'


def updateRouterConfigStats(network_config, router_config,\
                              network_stats, router_stats):
    # DSENT Interface

    # Config
    tech_model   = network_config['tech_model']
    num_vnet = network_config['num_vnet']
    flit_size_bits = network_config['flit_size_bits']

    frequency    = router_config['frequency']
    num_inports  = router_config['num_inports']
    num_outports = router_config['num_outports']
    vcs_per_vnet = router_config['vcs_per_vnet']
    buffers_per_ctrl_vc = network_config['buffers_per_ctrl_vc']
    buffers_per_data_vc = network_config['buffers_per_data_vc']

    # Stats
    sim_ticks           = network_stats['simTicks']
    buffer_writes       = router_stats['buffer_writes']
    buffer_reads        = router_stats['buffer_reads']
    sw_in_arb_activity  = router_stats['sw_in_arb_activity']
    sw_out_arb_activity = router_stats['sw_out_arb_activity']
    crossbar_activity   = router_stats['crossbar_activity']

    # Run DSENT (calls function in ext/dsent/interface.cc)
    print("\n|Router %s|" % router_config['router_id'])
 
    dsent.updateRouterConfigStats(tech_model,
                                  frequency,
                                  flit_size_bits,
                                  num_inports,
                                  num_outports,
                                  num_vnet,
                                  vcs_per_vnet,
                                  buffers_per_ctrl_vc,
                                  buffers_per_data_vc,
                                  sim_ticks,
                                  buffer_writes,
                                  buffer_reads,
                                  sw_in_arb_activity,
                                  sw_out_arb_activity,
                                  crossbar_activity)


## Compute the power consumed by the links
def updateLinkConfigStats(network_config, link_config, \
                     network_stats, link_stats):
 
    # DSENT Interface
    # Config
    tech_model  = network_config['tech_model']
    width_bits  = network_config['flit_size_bits']
    frequency   = link_config['frequency']
    length      = link_config['length']
    delay       = link_config['delay']

    # Stats
    sim_ticks   = network_stats['simTicks']
    activity    = link_stats['activity']
    # Run DSENT
    print("\n|All Links|")
    dsent.updateLinkConfigStats(tech_model,
                                frequency,
                                width_bits,
                                length,
                                delay,
                                sim_ticks,
                                activity)

# This script parses the config.ini and the stats.txt from a run and
# generates the power and the area of the on-chip network using DSENT
def main():
    parser = argparse.ArgumentParser(
        description="Parse gem5 config.ini and stats.txt to generate "
                    "on-chip network power and area using DSENT."
    )
    parser.add_argument(
        "gem5_root_dir",
        default=".",
        nargs="?",
        help="gem5 root directory. Default: '.'"
    )
    parser.add_argument(
        "sim_dir",
        default="m5out",
        nargs="?",
        help="Simulation output directory (e.g., m5out). Default: 'm5out'"
    )
    parser.add_argument(
        "router_config_file",
        default="ext/dsent/configs/garnet_router.cfg",
        nargs="?",
        help="DSENT router configuration file. Default: 'ext/dsent/configs/garnet_router.cfg'"
    )
    parser.add_argument(
        "link_config_file",
        default="ext/dsent/configs/garnet_link.cfg",
        nargs="?",
        help="DSENT link configuration file. Default: 'ext/dsent/configs/garnet_link.cfg'"
    )
    parser.add_argument(
        "tech_node",
        default="22",
        nargs="?",
        help="Technology node (e.g., 45, 32, 22, 11). Default: '22'"
    )
    parser.add_argument(
        "clock_period",
        type=int,
        nargs="?",
        default=None,
        help="Clock period in ps (optional). If not specified, it will be read "
             "from config.ini."
    )

    args = parser.parse_args()

    # Original argument checking based on sys.argv length, now handled by argparse or defaults
    # if len(sys.argv) < 6:
    #     print("\\nUsage: python ./" + sys.argv[0] + " <gem5 root directory>" \\
    #           " <simulation directory> " \\
    #           " <dsent router config file> <dsent link config file>" \\
    #           " <technology node>" \\
    #           " [<clock period in ps> (optional)]\\n"
    #           "Note: supported tech nodes: 45, 32, 22, and 11.\\n" \\
    #           "If clock period is not specified, it will be read from " \\
    #            "simulation directory/config.ini and can be different for "\\
    #            "each router and link")
    #     print("\\nExample: python ./" + sys.argv[0] + " . m5out " \\
    #           "ext/dsent/configs/garnet_router.cfg " \\
    #           "ext/dsent/configs/garnet_link.cfg " \\
    #           "32 500\\n" \\
    #           "This will model 500ps (2GHz) at 32nm")
    #     exit(-1)

    # tech = sys.argv[5] # Old
    tech = args.tech_node
    if not (tech == "45" or tech == "32" or tech == "22" or tech == "11"):
        print(
            "Error!! DSENT only supports 45nm (bulk), 32nm (bulk), "
            "22nm (bulk), and 11nm (tri-gate) models currently.\\n"
            "To model some other technology, add a model in "
            "ext/dsent/tech/tech_models.\\n"
            "To model photonic links, remove this warning, and update "
            # + sys.argv[0] + " to run DSENT with photonics.model with the " # Old
            + parser.prog + " to run DSENT with photonics.model with the " # New
            "appropriate configs and stats"
        )
        exit(-1)

    print(
        "WARNING: configuration files for DSENT and McPAT are separate. "
        "Changes made to one are not reflected in the other."
    )

    # config_file = "%s/%s/config.ini" % (sys.argv[1], sys.argv[2]) # Old
    # stats_file = "%s/%s/stats.txt" % (sys.argv[1], sys.argv[2]) # Old
    config_file = "%s/%s/config.ini" % (args.gem5_root_dir, args.sim_dir)
    stats_file = "%s/%s/stats.txt" % (args.gem5_root_dir, args.sim_dir)

    ### Parse Config File
    (all_config, network_config, routers, int_links, ext_links) = \
        parseConfig(config_file) # We need to pass tech_node to parseConfig or update it there

    # Update tech_model in network_config using args.tech_node
    tech_model_path = "ext/dsent/tech/tech_models/"
    if args.tech_node == "45" or args.tech_node == "32" or args.tech_node == "22":
        network_config['tech_model'] = tech_model_path + "Bulk" + args.tech_node + "LVT.model"
    elif args.tech_node == "11":
        network_config['tech_model'] = tech_model_path + "TG" + args.tech_node + "LVT.model"
    else:
        print(f"Unknown Tech Model: {args.tech_node}. Supported models: 45, 32, 22, 11")
        exit(-1)


    ### Parse Network Stats
    network_stats = parseNetworkStats(stats_file)

    ### Run DSENT
    # router_config_default = sys.argv[3] # Old
    # link_config_default = sys.argv[4] # Old
    router_config_default = args.router_config_file
    link_config_default = args.link_config_file

    if not os.path.isfile(router_config_default):
        print("ERROR: router config file '", router_config_default, "' not found")
        sys.exit(1)

    if not os.path.isfile(link_config_default):
        print("ERROR: link config file '", link_config_default, "' not found")
        sys.exit(1)

    ## Router Power and Area

    # Initialize DSENT with the router configuration file
    dsent.initialize(router_config_default)

    router_results = [] # To store results for each router

    # Update default configs for each router and run
    # Compute the power consumed by the routers
    for router in routers:
        router_config = getRouterConfig(all_config, router, \
            int_links, ext_links, args.clock_period)
        router_stats = parseRouterStats(stats_file, router)
        
        # Call the Python wrapper function, not the dsent C++ function directly.
        updateRouterConfigStats(network_config, router_config, \
                                network_stats, router_stats)

        with CaptureCOutput() as capture:
            dsent.run()
        
        router_results.append({
            "router_id": router_config['router_id'],
            "output": capture.captured_output
        })

    # Finalize DSENT
    dsent.finalize()

    ## Link Power

    # Initialize DSENT with a configuration file
    dsent.initialize(link_config_default)

    for link in int_links:
        link_config = getLinkConfig(all_config, link, args.clock_period)

    for link in ext_links:
        link_config = getLinkConfig(all_config, link, args.clock_period)


    # Update default configs file for links and run
    # (Stats file print total link activity rather than per link)
    # If per link power is required, garnet should print out activity
    # for each link, and updateLinkConfigStats should be called for each
    # link by uncommenting the link_stats and updateLinkConfigStats lines above
    sim_ticks = network_stats['simTicks']
    link_stats = parseLinkStats(stats_file, sim_ticks)
    updateLinkConfigStats(network_config, link_config, \
                          network_stats, link_stats)

    with CaptureCOutput() as capture:
        dsent.run()

    link_results = {
        "output": capture.captured_output
    }

    # Finalize DSENT
    dsent.finalize()

    # Dump the collected output to a file in the simulation directory
    output_filename = os.path.join(args.gem5_root_dir, args.sim_dir, "dsent_output.txt")
    with open(output_filename, "w") as f:
        print("\n--- Collected Router Results ---", file=f)
        for res in router_results:
            print(f"Router ID: {res['router_id']}", file=f)
            print(f"  Captured Output:\n{res['output']}", file=f)

        print("\n--- Collected Link Results ---", file=f)
        print(f"Captured Output:\n{link_results['output']}", file=f)
    print(f"\nDSENT output also dumped to {output_filename}")


    return router_results, link_results


if __name__ == "__main__":
    router_results, link_results = main()

    # Pretty print the results for verification when run as a standalone script
    print("\n--- Collected Router Results ---")
    for res in router_results:
        print(f"Router ID: {res['router_id']}")
        print(f"  Captured Output:\n{res['output']}")

    print("\n--- Collected Link Results ---")
    print(f"Captured Output:\n{link_results['output']}")
