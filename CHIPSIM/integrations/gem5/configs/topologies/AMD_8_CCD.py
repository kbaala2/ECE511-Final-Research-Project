from common import FileSystemConfig
from topologies.BaseTopology import SimpleTopology

from m5.objects import *
from m5.params import *
import sys

class AMD_8_CCD(SimpleTopology):
    description = "AMD_8_CCD"

    def __init__(self, controllers):
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        nodes = self.nodes #2x10 nodes: 8 CCDs + IOD + 1 DRAM 

        num_routers = options.num_cpus

        link_latency = options.link_latency
        router_latency = options.router_latency

        cntrls_per_router, remainder = divmod(len(nodes), num_routers)


        # width calculation:
        # for each router, assign max width, and play with link BW for each link to achieve desired BW
        ccd_read_wires = options.ccd_read_wires
        ccd_write_wires = options.ccd_write_wires
        dram_read_wires = options.dram_read_wires
        dram_write_wires = options.dram_write_wires

        max_width = max(ccd_read_wires, ccd_write_wires, dram_read_wires, dram_write_wires)

        routers = [Router(router_id=i, latency=router_latency, width=max_width) for i in range(num_routers)]
        network.routers = routers

        link_count = 0
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
                    width=max_width,
                )
            )
            link_count += 1
        network.ext_links = ext_links

        # int links

        num_ccd = 8
        int_links = []

        for i in range(num_ccd):
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[i],
                    dst_node=routers[8],  # IOD router
                    latency=link_latency,
                    width=ccd_write_wires,
                    src_serdes=True,
                    dst_serdes=True,
                )
            )
            link_count += 1

            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[8],  # IOD router
                    dst_node=routers[i],
                    latency=link_latency,
                    width=ccd_read_wires,
                    src_serdes=True,
                    dst_serdes=True,
                )
            )
            link_count += 1
        
        int_links.append(
            IntLink(
                link_id=link_count,
                src_node=routers[8],  # IOD router
                dst_node=routers[9],  # DRAM router
                latency=link_latency,
                width=dram_write_wires,
                src_serdes=True,
                dst_serdes=True,
            )
        )
        link_count += 1

        int_links.append(
            IntLink(
                link_id=link_count,
                src_node=routers[9],  # DRAM router
                dst_node=routers[8],  # IOD router
                latency=link_latency,
                width=dram_read_wires,
                src_serdes=False,
                dst_serdes=False,
            )
        )
        link_count += 1

        network.int_links = int_links

    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node(
                [i], MemorySize(options.mem_size) // options.num_cpus, i
            )
