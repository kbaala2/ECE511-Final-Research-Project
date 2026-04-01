from common import FileSystemConfig
from topologies.BaseTopology import SimpleTopology

from m5.objects import *
from m5.params import *

# Creates a generic Mesh assuming an equal number of cache
# and directory controllers.
# XY routing is enforced (using link weights)
# to guarantee deadlock freedom.


class SIMBA_PLUS_DRAM(SimpleTopology):
    description = "SIMBA_PLUS_DRAM"

    def __init__(self, controllers):
        self.nodes = controllers

    # Makes a generic mesh
    # assuming an equal number of cache and directory cntrls

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        nodes = self.nodes

        num_routers = options.num_cpus #60
        num_rows = 6 #6

        # default values for link latency and router latency.
        # Can be over-ridden on a per link/router basis
        link_latency = options.link_latency  # used by simple and garnet
        router_latency = options.router_latency  # only used by garnet

        # There must be an evenly divisible number of cntrls to routers
        # Also, obviously the number or rows must be <= the number of routers
        cntrls_per_router, remainder = divmod(len(nodes), num_routers)
        assert num_rows > 0 and num_rows <= num_routers
        num_columns = 6

        # Create the routers in the mesh
        routers = [
            Router(router_id=i, latency=router_latency)
            for i in range(num_routers)
        ]
        network.routers = routers

        # link counter to set unique link ids
        link_count = 0

        # Add all but the remainder nodes to the list of nodes to be uniformly
        # distributed across the network.
        network_nodes = []
        remainder_nodes = []
        for node_index in range(len(nodes)):
            if node_index < (len(nodes) - remainder):
                network_nodes.append(nodes[node_index])
            else:
                remainder_nodes.append(nodes[node_index])

        # Connect each node to the appropriate router
        ext_links = []
        # Connect each controller to a router (2:1 mapping assumed)
        for i, n in enumerate(nodes):
            cntrl_level, router_id = divmod(i, num_routers)
            assert cntrl_level < cntrls_per_router
            ext_links.append(
                ExtLink(
                    link_id=link_count,
                    ext_node=n,
                    int_node=routers[router_id],
                    latency=link_latency,
                )
            )
            link_count += 1
        network.ext_links = ext_links

        # Create the mesh links.
        int_links = []

        # East output to West input links (weight = 1)
        for row in range(num_rows):
            for col in range(num_columns):
                if col + 1 < num_columns:
                    east_out = col + (row * num_columns)
                    west_in = (col + 1) + (row * num_columns)
                    int_links.append(
                        IntLink(
                            link_id=link_count,
                            src_node=routers[east_out],
                            dst_node=routers[west_in],
                            latency=link_latency,
                            weight=1,
                        )
                    )
                    link_count += 1

                    int_links.append(
                        IntLink(
                            link_id=link_count,
                            src_node=routers[west_in],
                            dst_node=routers[east_out],
                            latency=link_latency,
                            weight=1,
                        )
                    )
                    link_count += 1

        # North output to South input links (weight = 2)
        # 0-11, 1-10, 2-9, 3-8, 4-7, 5-6 
        for col in range(6):
            weight = 2 if col==5 else 3
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[col],
                    dst_node=routers[11 - col],
                    latency=link_latency,
                    weight=weight,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[11 - col],
                    dst_node=routers[col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
        
        # 11-12, 10-13, 9-14, 8-15, 7-16, 6-17
        for col in range(6, 12):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[col],
                    dst_node=routers[23 - col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[23 - col],
                    dst_node=routers[col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1

        # 12-23, 13-22, 14-21, 15-20, 16-19, 17-18
        for col in range(12, 18):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[col],
                    dst_node=routers[35-col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[35-col],
                    dst_node=routers[col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1

        # 23-24, 22-25, 21-26, 20-27, 19-28, 18-29
        for col in range(18, 24):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[col],
                    dst_node=routers[47-col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[47-col],
                    dst_node=routers[col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1

        # 24-35, 25-34, 26-33, 27-32, 28-31, 29-30
        for col in range(24, 30):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[col],
                    dst_node=routers[59-col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[59-col],
                    dst_node=routers[col],
                    latency=link_latency,
                    weight=2,
                )
            )
            link_count += 1
        
        # dram 36-41 connected to chiplet 0-5
        for i in range(36, 42):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[i],
                    dst_node=routers[i-36],  # chiplet 0 to 5
                    latency=link_latency,
                )
            )
            link_count += 1

            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[i-36],  # chiplet 5 to 0
                    dst_node=routers[i],
                    latency=link_latency,
                )
            )
            link_count += 1

        # right dram links
        # dram 42 to 47 connected to chiplet 5,6,17,18,29,30
        if True:
            int_links.append(   
                IntLink(
                    link_id=link_count,
                    src_node=routers[42],
                    dst_node=routers[5],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[5],
                    dst_node=routers[42],
                    latency=link_latency,
                )
            )
            link_count += 1

        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[6],
                    dst_node=routers[43],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[43],
                    dst_node=routers[6],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[17],
                    dst_node=routers[44],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[44],
                    dst_node=routers[17],
                    latency=link_latency,
                )
            )
            link_count += 1 
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[18],
                    dst_node=routers[45],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[45],
                    dst_node=routers[18],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[29],
                    dst_node=routers[46],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[46],
                    dst_node=routers[29],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[30],
                    dst_node=routers[47],
                    latency=link_latency,
                )
            )
            link_count += 1

            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[47],
                    dst_node=routers[30],
                    latency=link_latency,
                )
            )
            link_count += 1

        
        # bottom dram links
        # dram 48 to 53 connected to chiplet 30-35
        for i in range(48, 54):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[i],
                    dst_node=routers[i-18],  # chiplet 30 to 35
                    latency=link_latency,
                )
            )
            link_count += 1

            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[i-18],  # chiplet 30 to 35
                    dst_node=routers[i],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        # dram 54-35, dram 55-24, dram 56-23, dram 57-12, dram 58-11, dram 59-0
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[54],
                    dst_node=routers[35],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[35],
                    dst_node=routers[54],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[55],
                    dst_node=routers[24],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[24],
                    dst_node=routers[55],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[56],
                    dst_node=routers[23],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[23],
                    dst_node=routers[56],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[57],
                    dst_node=routers[12],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[12],
                    dst_node=routers[57],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[58],
                    dst_node=routers[11],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[11],
                    dst_node=routers[58],
                    latency=link_latency,
                )
            )
            link_count += 1
        
        if True:
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[59],
                    dst_node=routers[0],
                    latency=link_latency,
                )
            )
            link_count += 1
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[0],
                    dst_node=routers[59],
                    latency=link_latency,
                )
            )
            link_count += 1

        network.int_links = int_links

    # Register nodes with filesystem
    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node(
                [i], MemorySize(options.mem_size) // options.num_cpus, i
            )
