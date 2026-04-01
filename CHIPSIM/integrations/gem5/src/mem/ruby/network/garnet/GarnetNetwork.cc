/*
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * Copyright (c) 2008 Princeton University
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "mem/ruby/network/garnet/GarnetNetwork.hh"

#include <cassert>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <tuple>

#include "base/cast.hh"
#include "base/compiler.hh"
#include "base/logging.hh"
#include "debug/RubyNetwork.hh"
#include "mem/ruby/common/NetDest.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/garnet/CommonTypes.hh"
#include "mem/ruby/network/garnet/CreditLink.hh"
#include "mem/ruby/network/garnet/GarnetLink.hh"
#include "mem/ruby/network/garnet/NetworkInterface.hh"
#include "mem/ruby/network/garnet/NetworkLink.hh"
#include "mem/ruby/network/garnet/Router.hh"
#include "mem/ruby/system/RubySystem.hh"
#include "sim/sim_exit.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

/*
 * GarnetNetwork sets up the routers and links and collects stats.
 * Default parameters (GarnetNetwork.py) can be overwritten from command line
 * (see configs/network/Network.py)
 */

GarnetNetwork::GarnetNetwork(const Params &p)
    : Network(p), Consumer(this)
{
    m_num_rows = p.num_rows;
    m_ni_flit_size = p.ni_flit_size;
    m_max_vcs_per_vnet = 0;
    m_buffers_per_data_vc = p.buffers_per_data_vc;
    m_buffers_per_ctrl_vc = p.buffers_per_ctrl_vc;
    m_routing_algorithm = p.routing_algorithm;
    m_next_packet_id = 0;

    m_enable_fault_model = p.enable_fault_model;
    if (m_enable_fault_model)
        fault_model = p.fault_model;

    m_trace_enable = p.trace_enable;
    m_trace_filename = p.trace_file;
    m_trace_max_packets = p.trace_max_packets;

    m_vnet_type.resize(m_virtual_networks);

    for (int i = 0 ; i < m_virtual_networks ; i++) {
        if (m_vnet_type_names[i] == "response")
            m_vnet_type[i] = DATA_VNET_; // carries data (and ctrl) packets
        else
            m_vnet_type[i] = CTRL_VNET_; // carries only ctrl packets
    }

    const char *trace_filename; //[100];
    trace_filename = m_trace_filename.c_str();
    //strcpy(trace_filename, m_trace_filename);
    char  command_string[512];
  
    //sprintf(command_string,"gunzip -c %s", trace_filename);
    if (endsWith(trace_filename, ".gz")) {
        sprintf(command_string,"gunzip -c %s", trace_filename);
    } else {
        sprintf(command_string,"tail -f %s", trace_filename);
    }

    if (( tracefile= popen(command_string, "r")) == NULL){
        printf("Command string is %s\n", command_string);
    }

    // Read first thirteen lines of the trace
    int count = 1;
    char trstring [1000];

    while (count > 0) {
        if ( fgets (trstring , 1000, tracefile) != NULL ) {
            count--;
            //puts (trstring);
        } else {
            break;
        }
    }
    printf("Done reading first 13 lines\n");

/*
    char trstring [1000];
    bool done = false;

    // Read trace up to point when packets start
    while (!done) {
        if ( fgets (trstring , 1000, tracefile) != NULL ) {
            puts (trstring);

            // Step 2: Get tokens

            // array to store memory addresses of the tokens in buf
            const char* first_token;

            first_token = strtok(trstring, " "); // first token
            std::string trace_start = "info";
            
            if (strcmp(first_token, trace_start.c_str())) {
                done = true;
            }
        } else {
            done = true;
        }
    }
*/
    trace_num_packets_injected = 0;
    trace_num_flits_injected = 0;
    trace_num_flits_received = 0;
    
    trace_start_time = 0;

    // Initialize next packet
    trace_next_packet.valid = false;
    trace_next_packet.time = Cycles(0);
    trace_next_packet.src_id = -1;
    trace_next_packet.dest_id = -1;

    /*
    // Trace Read
    ifstream fin;
    fin.open("data.txt");
    if (!fin.good())
        exit(0);

    // read each line of the file
    while (!fin.eof())
    {
        // read an entire line into memory
        char buf[MAX_CHARS_PER_LINE];
        fin.getline(buf, MAX_CHARS_PER_LINE);

        // parse the line into blank-delimited tokens
        int n = 0; // loop idex

        // array to store memory addresses of the tokens in buf
        const char* token[MAX_TOKENS_PER_LINE] = {}; // initialize to 0

        // parse the line
        token[0] = strtok(buf, DELIMITER); // first token
        if (token[0]) // zero if line is blank
        {
            for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
            {
                token[n] = strtok(0, DELIMITER); // subsequent tokens
                if (!token[n]) break; // no more tokens
            }
        }

        // process (print) the tokens
        for (int i = 0; i < n; i++) // n = #of tokens
            DPRINTF(NetworkTrace, "Token[%d] = %s\n", i, token[i]);
    }
    */

    scheduleWakeupAbsolute(curCycle() + Cycles(1));

    // record the routers
    for (std::vector<BasicRouter*>::const_iterator i =  p.routers.begin();
         i != p.routers.end(); ++i) {
        Router* router = safe_cast<Router*>(*i);
        m_routers.push_back(router);

        // initialize the router's network pointers
        router->init_net_ptr(this);
    }

    // record the network interfaces
    for (std::vector<ClockedObject*>::const_iterator i = p.netifs.begin();
         i != p.netifs.end(); ++i) {
        NetworkInterface *ni = safe_cast<NetworkInterface *>(*i);
        m_nis.push_back(ni);
        ni->init_net_ptr(this);
    }

    // Print Garnet version
    inform("Garnet version %s\n", garnetVersion);
}

bool 
GarnetNetwork::endsWith(const char* str, const char* suffix) {
    size_t len1 = std::strlen(str);
    size_t len2 = std::strlen(suffix);
    return (len1 >= len2) && (std::strcmp(str + len1 - len2, suffix) == 0);
}

void
GarnetNetwork::scheduleWakeupAbsolute(Cycles time)
{
    // wake up at time
    scheduleEventAbsolute(cyclesToTicks(time));
}

void
GarnetNetwork::wakeup()
{
    DPRINTF(NetworkTrace, "GarnetNetwork woke up at time: %lld\n", curCycle());

    //m_nis[0]->enqueueTracePacket(0, 0, 4, 4, 0, 5);
    //m_nis[0]->enqueueTracePacket(2, 2, 10, 10, 0, 1);

    // Only wakes up if time stamp of next packet matches current time
    if (trace_next_packet.valid) {
        assert(trace_next_packet.time == curCycle());
    
        // Insert this packet
        DPRINTF(NetworkTrace,
            "GarnetNetwork injecting packet at time %lld from NI %d "
            "router %d to NI %d router %d in vnet %d with num flits %d\n",
            trace_next_packet.time, trace_next_packet.src_id,
            trace_next_packet.src_router_id, trace_next_packet.dest_id,
            trace_next_packet.dest_router_id, trace_next_packet.vnet,
            trace_next_packet.num_flits);

        m_nis[trace_next_packet.src_id]->
             enqueueTracePacket(trace_next_packet);

        // stats
        trace_num_packets_injected++;
        trace_num_flits_injected =
            trace_num_flits_injected + trace_next_packet.num_flits;
    }
    // Read next packet from trace

    #define MAX_TOKENS_PER_LINE 100
    #define DELIMITER " :"
    char trstring [MAX_TOKENS_PER_LINE];

    // Step 1: Read line
    if ( fgets (trstring , MAX_TOKENS_PER_LINE, tracefile) != NULL ) {
        // Print string:
        //puts (trstring);

        // Step 2: Get tokens

        // array to store memory addresses of the tokens in buf
        const char* token[MAX_TOKENS_PER_LINE] = {}; // initialize to 0

        int n = 0; // loop index
        token[0] = strtok(trstring, DELIMITER); // first token
        if (token[0]) // zero if line is blank
        {
            for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
            {
                token[n] = strtok(0, DELIMITER); // subsequent tokens
                if (!token[n]) break; // no more tokens
            }
        } else {
            return;
        }

        if (trace_num_packets_injected == 0) {
            trace_start_time = atoi(token[0]);
        }

        int next_packet_time = atoi(token[0]);
        // int next_packet_time = atoi(token[2]) - trace_start_time + 2;

        // Step 3: Extract packet
        trace_next_packet.time = Cycles(next_packet_time);
        trace_next_packet.src_id = atoi(token[1]);
        trace_next_packet.src_router_id = atoi(token[2]);
        trace_next_packet.dest_id = atoi(token[3]);
        trace_next_packet.dest_router_id = atoi(token[4]);
        trace_next_packet.vnet = atoi(token[5]);
        trace_next_packet.num_flits = atoi(token[6]);
        
        // Parse traffic source identification
        // If tokens 7, 8, 9 exist, use them; otherwise default to 0
        if (n > 7) {
            trace_next_packet.network_idx = atoi(token[7]);
        } else {
            trace_next_packet.network_idx = 0;
        }
        
        if (n > 8) {
            trace_next_packet.input_idx = atoi(token[8]);
        } else {
            trace_next_packet.input_idx = 0;
        }
        
        if (n > 9) {
            trace_next_packet.phase_id = atoi(token[9]);
        } else {
            trace_next_packet.phase_id = 0;
        }

        if (trace_next_packet.src_id >= m_nis.size()) {
            // Invalid packet
            // This is a DMA packet
            //trace_next_packet.valid = false;

            // Send from NI 0 instead
            trace_next_packet.src_id = 0;
            trace_next_packet.src_router_id = 0;
            trace_next_packet.valid = true;

        } else if (trace_next_packet.dest_id >= m_nis.size()) {

            // Invalid packet
            // This is a DMA packet
            //trace_next_packet.valid = false;

            // send to NI 0 instead
            trace_next_packet.dest_id = 0;
            trace_next_packet.dest_router_id = 0;
            trace_next_packet.valid = true;

        } else {
            trace_next_packet.valid = true;
        }

        // process (print) the tokens
        for (int i = 0; i < n; i++) // n = #of tokens
            DPRINTF(NetworkTrace, "Token[%d] = %s\n", i, token[i]);

        if (m_trace_max_packets < 0 || trace_num_packets_injected <= m_trace_max_packets) {
            if (trace_next_packet.time == curCycle())
                wakeup(); // inject another packet
            else
                scheduleWakeupAbsolute(trace_next_packet.time);
        }
    } else {
        if(feof(tracefile)){
            end_simulation = true;
        } else {
            perror("Error reading trace file");
        }
    }
}

void
GarnetNetwork::increment_trace_flits_received()
{
    trace_num_flits_received++;

    DPRINTF(NetworkTrace, "Num flits injected = %d, Num flits received = %d\n",
        trace_num_flits_injected, trace_num_flits_received);

    if ((end_simulation && trace_num_flits_received == trace_num_flits_injected) || 
        (m_trace_max_packets >= 0 && trace_num_packets_injected > m_trace_max_packets &&
        trace_num_flits_received == trace_num_flits_injected)) {

            exitSimLoop("Network Trace Simulation Complete.");
    }
}

void
GarnetNetwork::init()
{
    Network::init();

    for (int i=0; i < m_nodes; i++) {
        m_nis[i]->addNode(m_toNetQueues[i], m_fromNetQueues[i]);
    }

    // The topology pointer should have already been initialized in the
    // parent network constructor
    assert(m_topology_ptr != NULL);
    m_topology_ptr->createLinks(this);

    // Initialize topology specific parameters
    if (getNumRows() > 0) {
        // Only for Mesh topology
        // m_num_rows and m_num_cols are only used for
        // implementing XY or custom routing in RoutingUnit.cc
        m_num_rows = getNumRows();
        m_num_cols = m_routers.size() / m_num_rows;
        assert(m_num_rows * m_num_cols == m_routers.size());
    } else {
        m_num_rows = -1;
        m_num_cols = -1;
    }

    // FaultModel: declare each router to the fault model
    if (isFaultModelEnabled()) {
        for (std::vector<Router*>::const_iterator i= m_routers.begin();
             i != m_routers.end(); ++i) {
            Router* router = safe_cast<Router*>(*i);
            [[maybe_unused]] int router_id =
                fault_model->declare_router(router->get_num_inports(),
                                            router->get_num_outports(),
                                            router->get_vc_per_vnet(),
                                            getBuffersPerDataVC(),
                                            getBuffersPerCtrlVC());
            assert(router_id == router->get_id());
            router->printAggregateFaultProbability(std::cout);
            router->printFaultVector(std::cout);
        }
    }
}

/*
 * This function creates a link from the Network Interface (NI)
 * into the Network.
 * It creates a Network Link from the NI to a Router and a Credit Link from
 * the Router to the NI
*/

void
GarnetNetwork::makeExtInLink(NodeID global_src, SwitchID dest, BasicLink* link,
                             std::vector<NetDest>& routing_table_entry)
{
    NodeID local_src = getLocalNodeID(global_src);
    assert(local_src < m_nodes);

    GarnetExtLink* garnet_link = safe_cast<GarnetExtLink*>(link);

    // GarnetExtLink is bi-directional
    NetworkLink* net_link = garnet_link->m_network_links[LinkDirection_In];
    net_link->setType(EXT_IN_);
    CreditLink* credit_link = garnet_link->m_credit_links[LinkDirection_In];

    m_networklinks.push_back(net_link);
    m_creditlinks.push_back(credit_link);

    PortDirection dst_inport_dirn = "Local";

    m_max_vcs_per_vnet = std::max(m_max_vcs_per_vnet,
                             m_routers[dest]->get_vc_per_vnet());

    /*
     * We check if a bridge was enabled at any end of the link.
     * The bridge is enabled if either of clock domain
     * crossing (CDC) or Serializer-Deserializer(SerDes) unit is
     * enabled for the link at each end. The bridge encapsulates
     * the functionality for both CDC and SerDes and is a Consumer
     * object similiar to a NetworkLink.
     *
     * If a bridge was enabled we connect the NI and Routers to
     * bridge before connecting the link. Example, if an external
     * bridge is enabled, we would connect:
     * NI--->NetworkBridge--->GarnetExtLink---->Router
     */
    if (garnet_link->extBridgeEn) {
        DPRINTF(RubyNetwork, "Enable external bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->extNetBridge[LinkDirection_In];
        m_nis[local_src]->
        addOutPort(n_bridge,
                   garnet_link->extCredBridge[LinkDirection_In],
                   dest, m_routers[dest]->get_vc_per_vnet());
        m_networkbridges.push_back(n_bridge);
    } else {
        m_nis[local_src]->addOutPort(net_link, credit_link, dest,
            m_routers[dest]->get_vc_per_vnet());
    }

    if (garnet_link->intBridgeEn) {
        DPRINTF(RubyNetwork, "Enable internal bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->intNetBridge[LinkDirection_In];
        m_routers[dest]->
            addInPort(dst_inport_dirn,
                      n_bridge,
                      garnet_link->intCredBridge[LinkDirection_In]);
        m_networkbridges.push_back(n_bridge);
    } else {
        m_routers[dest]->addInPort(dst_inport_dirn, net_link, credit_link);
    }

}

/*
 * This function creates a link from the Network to a NI.
 * It creates a Network Link from a Router to the NI and
 * a Credit Link from NI to the Router
*/

void
GarnetNetwork::makeExtOutLink(SwitchID src, NodeID global_dest,
                              BasicLink* link,
                              std::vector<NetDest>& routing_table_entry)
{
    NodeID local_dest = getLocalNodeID(global_dest);
    assert(local_dest < m_nodes);
    assert(src < m_routers.size());
    assert(m_routers[src] != NULL);

    GarnetExtLink* garnet_link = safe_cast<GarnetExtLink*>(link);

    // GarnetExtLink is bi-directional
    NetworkLink* net_link = garnet_link->m_network_links[LinkDirection_Out];
    net_link->setType(EXT_OUT_);
    CreditLink* credit_link = garnet_link->m_credit_links[LinkDirection_Out];

    m_networklinks.push_back(net_link);
    m_creditlinks.push_back(credit_link);

    PortDirection src_outport_dirn = "Local";

    m_max_vcs_per_vnet = std::max(m_max_vcs_per_vnet,
                             m_routers[src]->get_vc_per_vnet());

    /*
     * We check if a bridge was enabled at any end of the link.
     * The bridge is enabled if either of clock domain
     * crossing (CDC) or Serializer-Deserializer(SerDes) unit is
     * enabled for the link at each end. The bridge encapsulates
     * the functionality for both CDC and SerDes and is a Consumer
     * object similiar to a NetworkLink.
     *
     * If a bridge was enabled we connect the NI and Routers to
     * bridge before connecting the link. Example, if an external
     * bridge is enabled, we would connect:
     * NI<---NetworkBridge<---GarnetExtLink<----Router
     */
    if (garnet_link->extBridgeEn) {
        DPRINTF(RubyNetwork, "Enable external bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->extNetBridge[LinkDirection_Out];
        m_nis[local_dest]->
            addInPort(n_bridge, garnet_link->extCredBridge[LinkDirection_Out]);
        m_networkbridges.push_back(n_bridge);
    } else {
        m_nis[local_dest]->addInPort(net_link, credit_link);
    }

    if (garnet_link->intBridgeEn) {
        DPRINTF(RubyNetwork, "Enable internal bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->intNetBridge[LinkDirection_Out];
        m_routers[src]->
            addOutPort(src_outport_dirn,
                       n_bridge,
                       routing_table_entry, link->m_weight,
                       garnet_link->intCredBridge[LinkDirection_Out],
                       m_routers[src]->get_vc_per_vnet());
        m_networkbridges.push_back(n_bridge);
    } else {
        m_routers[src]->
            addOutPort(src_outport_dirn, net_link,
                       routing_table_entry,
                       link->m_weight, credit_link,
                       m_routers[src]->get_vc_per_vnet());
    }
}

/*
 * This function creates an internal network link between two routers.
 * It adds both the network link and an opposite credit link.
*/

void
GarnetNetwork::makeInternalLink(SwitchID src, SwitchID dest, BasicLink* link,
                                std::vector<NetDest>& routing_table_entry,
                                PortDirection src_outport_dirn,
                                PortDirection dst_inport_dirn)
{
    GarnetIntLink* garnet_link = safe_cast<GarnetIntLink*>(link);

    // GarnetIntLink is unidirectional
    NetworkLink* net_link = garnet_link->m_network_link;
    net_link->setType(INT_);
    CreditLink* credit_link = garnet_link->m_credit_link;

    m_networklinks.push_back(net_link);
    m_creditlinks.push_back(credit_link);

    m_max_vcs_per_vnet = std::max(m_max_vcs_per_vnet,
                             std::max(m_routers[dest]->get_vc_per_vnet(),
                             m_routers[src]->get_vc_per_vnet()));

    /*
     * We check if a bridge was enabled at any end of the link.
     * The bridge is enabled if either of clock domain
     * crossing (CDC) or Serializer-Deserializer(SerDes) unit is
     * enabled for the link at each end. The bridge encapsulates
     * the functionality for both CDC and SerDes and is a Consumer
     * object similiar to a NetworkLink.
     *
     * If a bridge was enabled we connect the NI and Routers to
     * bridge before connecting the link. Example, if a source
     * bridge is enabled, we would connect:
     * Router--->NetworkBridge--->GarnetIntLink---->Router
     */
    if (garnet_link->dstBridgeEn) {
        DPRINTF(RubyNetwork, "Enable destination bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->dstNetBridge;
        m_routers[dest]->addInPort(dst_inport_dirn, n_bridge,
                                   garnet_link->dstCredBridge);
        m_networkbridges.push_back(n_bridge);
    } else {
        m_routers[dest]->addInPort(dst_inport_dirn, net_link, credit_link);
    }

    if (garnet_link->srcBridgeEn) {
        DPRINTF(RubyNetwork, "Enable source bridge for %s\n",
            garnet_link->name());
        NetworkBridge *n_bridge = garnet_link->srcNetBridge;
        m_routers[src]->
            addOutPort(src_outport_dirn, n_bridge,
                       routing_table_entry,
                       link->m_weight, garnet_link->srcCredBridge,
                       m_routers[dest]->get_vc_per_vnet());
        m_networkbridges.push_back(n_bridge);
    } else {
        m_routers[src]->addOutPort(src_outport_dirn, net_link,
                        routing_table_entry,
                        link->m_weight, credit_link,
                        m_routers[dest]->get_vc_per_vnet());
    }
}

// Total routers in the network
int
GarnetNetwork::getNumRouters()
{
    return m_routers.size();
}

// Get ID of router connected to a NI.
int
GarnetNetwork::get_router_id(int global_ni, int vnet)
{
    NodeID local_ni = getLocalNodeID(global_ni);

    return m_nis[local_ni]->get_router_id(vnet);
}

void
GarnetNetwork::magicSend(int ni, int vnet, MsgPtr msg_ptr)
{
    m_nis[ni]->magicReceive(vnet, msg_ptr);
}

void
GarnetNetwork::regStats()
{
    Network::regStats();

    // Packets
    m_packets_received
        .init(m_virtual_networks)
        .name(name() + ".packets_received")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    m_packets_injected
        .init(m_virtual_networks)
        .name(name() + ".packets_injected")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    m_packet_network_latency
        .init(m_virtual_networks)
        .name(name() + ".packet_network_latency")
        .flags(statistics::oneline)
        ;

    m_packet_queueing_latency
        .init(m_virtual_networks)
        .name(name() + ".packet_queueing_latency")
        .flags(statistics::oneline)
        ;

    for (int i = 0; i < m_virtual_networks; i++) {
        m_packets_received.subname(i, csprintf("vnet-%i", i));
        m_packets_injected.subname(i, csprintf("vnet-%i", i));
        m_packet_network_latency.subname(i, csprintf("vnet-%i", i));
        m_packet_queueing_latency.subname(i, csprintf("vnet-%i", i));
    }

    m_avg_packet_vnet_latency
        .name(name() + ".average_packet_vnet_latency")
        .flags(statistics::oneline);
    m_avg_packet_vnet_latency =
        m_packet_network_latency / m_packets_received;

    m_avg_packet_vqueue_latency
        .name(name() + ".average_packet_vqueue_latency")
        .flags(statistics::oneline);
    m_avg_packet_vqueue_latency =
        m_packet_queueing_latency / m_packets_received;

    m_avg_packet_network_latency
        .name(name() + ".average_packet_network_latency");
    m_avg_packet_network_latency =
        sum(m_packet_network_latency) / sum(m_packets_received);

    m_avg_packet_queueing_latency
        .name(name() + ".average_packet_queueing_latency");
    m_avg_packet_queueing_latency
        = sum(m_packet_queueing_latency) / sum(m_packets_received);

    m_avg_packet_latency
        .name(name() + ".average_packet_latency");
    m_avg_packet_latency
        = m_avg_packet_network_latency + m_avg_packet_queueing_latency;

    m_total_main_layer_pkts
        .name(name() + ".total_main_layer_pkts")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    m_packet_max_latency
        .name(name() + ".main_layer_latency");

    // Pre-register layer latency stats for all possible combinations
    // These limits should cover most use cases while keeping memory reasonable
    // If you need more, increase these constants
    const int MAX_NETWORKS = 20;  // Maximum number of networks to track
    const int MAX_INPUTS = 20;    // Maximum number of inputs per network
    const int MAX_PHASES = 500;   // Maximum number of phases per network
    
    for (int net = 0; net < MAX_NETWORKS; net++) {
        for (int inp = 0; inp < MAX_INPUTS; inp++) {
            for (int phase = 0; phase < MAX_PHASES; phase++) {
                auto key = std::make_tuple(net, inp, phase);
                std::string stat_name = name() + ".phase_latency_" + 
                                        std::to_string(net) + "_" +
                                        std::to_string(inp) + "_" +
                                        std::to_string(phase);
                
                m_phase_latency_stats[key] = new statistics::Scalar();
                m_phase_latency_stats[key]->name(stat_name);
                m_phase_latency_stats[key]->flags(statistics::nozero);
            }
        }
    }

    // Flits
    m_flits_received
        .init(m_virtual_networks)
        .name(name() + ".flits_received")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    m_flits_injected
        .init(m_virtual_networks)
        .name(name() + ".flits_injected")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    m_flit_network_latency
        .init(m_virtual_networks)
        .name(name() + ".flit_network_latency")
        .flags(statistics::oneline)
        ;

    m_flit_queueing_latency
        .init(m_virtual_networks)
        .name(name() + ".flit_queueing_latency")
        .flags(statistics::oneline)
        ;

    for (int i = 0; i < m_virtual_networks; i++) {
        m_flits_received.subname(i, csprintf("vnet-%i", i));
        m_flits_injected.subname(i, csprintf("vnet-%i", i));
        m_flit_network_latency.subname(i, csprintf("vnet-%i", i));
        m_flit_queueing_latency.subname(i, csprintf("vnet-%i", i));
    }

    m_avg_flit_vnet_latency
        .name(name() + ".average_flit_vnet_latency")
        .flags(statistics::oneline);
    m_avg_flit_vnet_latency = m_flit_network_latency / m_flits_received;

    m_avg_flit_vqueue_latency
        .name(name() + ".average_flit_vqueue_latency")
        .flags(statistics::oneline);
    m_avg_flit_vqueue_latency =
        m_flit_queueing_latency / m_flits_received;

    m_avg_flit_network_latency
        .name(name() + ".average_flit_network_latency");
    m_avg_flit_network_latency =
        sum(m_flit_network_latency) / sum(m_flits_received);

    m_avg_flit_queueing_latency
        .name(name() + ".average_flit_queueing_latency");
    m_avg_flit_queueing_latency =
        sum(m_flit_queueing_latency) / sum(m_flits_received);

    m_avg_flit_latency
        .name(name() + ".average_flit_latency");
    m_avg_flit_latency =
        m_avg_flit_network_latency + m_avg_flit_queueing_latency;


    // Hops
    m_avg_hops.name(name() + ".average_hops");
    m_avg_hops = m_total_hops / sum(m_flits_received);

    // Links
    m_total_ext_in_link_utilization
        .name(name() + ".ext_in_link_utilization");
    m_total_ext_out_link_utilization
        .name(name() + ".ext_out_link_utilization");
    m_total_int_link_utilization
        .name(name() + ".int_link_utilization");
    m_average_link_utilization
        .name(name() + ".avg_link_utilization");
    m_average_vc_load
        .init(m_virtual_networks * m_max_vcs_per_vnet)
        .name(name() + ".avg_vc_load")
        .flags(statistics::pdf | statistics::total | statistics::nozero |
            statistics::oneline)
        ;

    // Traffic distribution
    for (int source = 0; source < m_routers.size(); ++source) {
        m_data_traffic_distribution.push_back(
            std::vector<statistics::Scalar *>());
        m_ctrl_traffic_distribution.push_back(
            std::vector<statistics::Scalar *>());

        for (int dest = 0; dest < m_routers.size(); ++dest) {
            statistics::Scalar *data_packets = new statistics::Scalar();
            statistics::Scalar *ctrl_packets = new statistics::Scalar();

            data_packets->name(name() + ".data_traffic_distribution." + "n" +
                    std::to_string(source) + "." + "n" + std::to_string(dest));
            m_data_traffic_distribution[source].push_back(data_packets);

            ctrl_packets->name(name() + ".ctrl_traffic_distribution." + "n" +
                    std::to_string(source) + "." + "n" + std::to_string(dest));
            m_ctrl_traffic_distribution[source].push_back(ctrl_packets);
        }
    }
}

void
GarnetNetwork::collateStats()
{
    RubySystem *rs = params().ruby_system;
    double time_delta = double(curCycle() - rs->getStartCycle());

    for (int i = 0; i < m_networklinks.size(); i++) {
        link_type type = m_networklinks[i]->getType();
        int activity = m_networklinks[i]->getLinkUtilization();

        if (type == EXT_IN_)
            m_total_ext_in_link_utilization += activity;
        else if (type == EXT_OUT_)
            m_total_ext_out_link_utilization += activity;
        else if (type == INT_)
            m_total_int_link_utilization += activity;

        m_average_link_utilization +=
            (double(activity) / time_delta);

        std::vector<unsigned int> vc_load = m_networklinks[i]->getVcLoad();
        for (int j = 0; j < vc_load.size(); j++) {
            m_average_vc_load[j] += ((double)vc_load[j] / time_delta);
        }
    }

    // Ask the routers to collate their statistics
    for (int i = 0; i < m_routers.size(); i++) {
        m_routers[i]->collateStats();
    }
    
    // Set per-phase latency statistics values
    for (const auto& entry : m_packet_phase_latencies) {
        auto key = entry.first;
        const auto& latencies = entry.second;
        
        if (!latencies.empty()) {
            // Calculate max latency for this phase
            Tick max_latency = *std::max_element(latencies.begin(), latencies.end());
            
            // The stat should already be registered in regStats()
            if (m_phase_latency_stats.find(key) != m_phase_latency_stats.end()) {
                // Set the value
                *(m_phase_latency_stats[key]) = max_latency;
            } else {
                // Warn if we're trying to set a stat that wasn't pre-registered
                warn("Phase latency stat for network %d, input %d, phase %d "
                     "was not pre-registered. Increase MAX_NETWORKS, MAX_INPUTS, "
                     "or MAX_PHASES in GarnetNetwork::regStats() if needed.",
                     std::get<0>(key), std::get<1>(key), std::get<2>(key));
            }
        }
    }
}

void
GarnetNetwork::resetStats()
{
    for (int i = 0; i < m_routers.size(); i++) {
        m_routers[i]->resetStats();
    }
    for (int i = 0; i < m_networklinks.size(); i++) {
        m_networklinks[i]->resetStats();
    }
    for (int i = 0; i < m_creditlinks.size(); i++) {
        m_creditlinks[i]->resetStats();
    }
}

void
GarnetNetwork::print(std::ostream& out) const
{
    out << "[GarnetNetwork]";
}

void
GarnetNetwork::update_traffic_distribution(RouteInfo route)
{
    int src_node = route.src_router;
    int dest_node = route.dest_router;
    int vnet = route.vnet;

    if (m_vnet_type[vnet] == DATA_VNET_)
        (*m_data_traffic_distribution[src_node][dest_node])++;
    else
        (*m_ctrl_traffic_distribution[src_node][dest_node])++;
}

bool
GarnetNetwork::functionalRead(Packet *pkt, WriteMask &mask)
{
    bool read = false;
    for (unsigned int i = 0; i < m_routers.size(); i++) {
        if (m_routers[i]->functionalRead(pkt, mask))
            read = true;
    }

    for (unsigned int i = 0; i < m_nis.size(); ++i) {
        if (m_nis[i]->functionalRead(pkt, mask))
            read = true;
    }

    for (unsigned int i = 0; i < m_networklinks.size(); ++i) {
        if (m_networklinks[i]->functionalRead(pkt, mask))
            read = true;
    }

    for (unsigned int i = 0; i < m_networkbridges.size(); ++i) {
        if (m_networkbridges[i]->functionalRead(pkt, mask))
            read = true;
    }

    return read;
}

uint32_t
GarnetNetwork::functionalWrite(Packet *pkt)
{
    uint32_t num_functional_writes = 0;

    for (unsigned int i = 0; i < m_routers.size(); i++) {
        num_functional_writes += m_routers[i]->functionalWrite(pkt);
    }

    for (unsigned int i = 0; i < m_nis.size(); ++i) {
        num_functional_writes += m_nis[i]->functionalWrite(pkt);
    }

    for (unsigned int i = 0; i < m_networklinks.size(); ++i) {
        num_functional_writes += m_networklinks[i]->functionalWrite(pkt);
    }

    return num_functional_writes;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
