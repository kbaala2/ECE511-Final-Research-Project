from common import FileSystemConfig
from topologies.BaseTopology import SimpleTopology

from m5.objects import *
from m5.params import *
import math
import sys
#path to your yaml package
#sys.path.insert(0, "/disk/user/project_dir/gem5/build/Garnet_standalone/python-packages")
import yaml

class AnyNET_XY(SimpleTopology):
    description = "AnyNET_XY"

    def __init__(self, controllers):
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        nodes = self.nodes
        link_latency = options.link_latency
        router_latency = options.router_latency
        num_routers = options.num_cpus

        routers = [Router(router_id=i, latency=router_latency) for i in range(num_routers)]
        network.routers = routers

        link_count = 0
        ext_links = []
        
        # Connect each controller to a router (2:1 mapping assumed)
        for i, node in enumerate(nodes):
            router_id = i % len(routers)  # wraps around if more nodes than routers
            ext_links.append(
                ExtLink(
                    link_id=link_count,
                    ext_node=node,
                int_node=routers[router_id],
                latency=link_latency,
                )
            )
            link_count += 1

        # Read allowed inter-router connections from YAML
        allowed_set = set()
        with open(options.config_file, "r") as f:
            allowed_links = yaml.safe_load(f).get("connections", {})
            for chiplet, connect_list in allowed_links.items():
                for conn in connect_list:
                    allowed_set.add(tuple(conn))

        # Determine grid dimensions (assumes regular mesh-style layout)
        num_rows = getattr(options, "mesh_rows", None)
        if num_rows is None or num_rows <= 0:
            num_rows = int(math.sqrt(num_routers))
            if num_rows == 0 or num_rows * num_rows != num_routers:
                raise ValueError(
                    "Unable to infer mesh_rows for AnyNET topology; please specify --mesh-rows"
                )
        num_cols = num_routers // num_rows
        if num_rows * num_cols != num_routers or num_cols == 0:
            raise ValueError(
                f"Invalid mesh dimensions derived for AnyNET topology: rows={num_rows}, cols={num_cols}"
            )

        int_links = []

        def append_link(src, dst, outport, inport, weight):
            nonlocal link_count
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[src],
                    dst_node=routers[dst],
                    src_outport=outport,
                    dst_inport=inport,
                    latency=link_latency,
                    weight=weight,
                )
            )
            link_count += 1

        # Create internal links based on YAML
        for (src, dst) in allowed_set:
            src_row, src_col = divmod(src, num_cols)
            dst_row, dst_col = divmod(dst, num_cols)

            if src_row == dst_row:
                link_weight = 1  # Horizontal link (east/west)
            elif src_col == dst_col:
                link_weight = 2  # Vertical link (north/south)
            else:
                link_weight = 1  # Diagonal or irregular link; default to 1

            append_link(src, dst, "Link", "Link", link_weight)

        network.ext_links = ext_links
        network.int_links = int_links

    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node(
                [i], MemorySize(options.mem_size) // options.num_cpus, i
            )
