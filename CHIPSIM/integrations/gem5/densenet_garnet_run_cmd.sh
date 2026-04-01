#!/bin/bash

# --debug-flags=NetworkTrace  \

bits=24

case $bits in
        0)
        max_pkts=254975
        ;;
        4)
        max_pkts=223103
        ;;
        8)
        max_pkts=191231
        ;;
        12)
        max_pkts=159359
        ;;
        16)
        max_pkts=127487
        ;;
        20)
        max_pkts=95615
        ;;
        22)
        max_pkts=79679
        ;;
        24)
        max_pkts=63743
        ;;
        *)
        echo "Invalid bits"
        exit 1
        ;;
esac

./build/Garnet_standalone/gem5.opt \
        configs/example/garnet_synth_traffic.py \
        --num-cpus=120 --num-dirs=120 --topology=Mesh_XY --mesh-rows=12 \
        --sim-cycles=50000000 --injectionrate=0 --network-trace-enable \
        --network-trace-file="1_757_files/DenseNet121/DenseNet121_bits_"$bits"_garnet_traffic.txt.gz" \
        --network-trace-max-packets=$max_pkts 

sleep 1


echo "power run"
python util/on-chip-network-power-area.py .  \
        m5out \
        ext/dsent/configs/garnet_router.cfg \
        ext/dsent/configs/garnet_link.cfg \
        32 1000 > log_power