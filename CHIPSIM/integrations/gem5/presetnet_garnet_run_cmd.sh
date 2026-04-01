#!/bin/bash

# --debug-flags=NetworkTrace  \

bits=24

case $bits in
        0)
        max_pkts=24847
        ;;
        4)
        max_pkts=21741
        ;;
        8)
        max_pkts=18635
        ;;
        12)
        max_pkts=15529
        ;;
        16)
        max_pkts=12423
        ;;
        20)
        max_pkts=9317
        ;;
        22)
        max_pkts=7764
        ;;
        24)
        max_pkts=6211
        ;;
        *)
        echo "Invalid bits"
        exit 1
        ;;
esac

./build/Garnet_standalone/gem5.opt \
        configs/example/garnet_synth_traffic.py \
        --num-cpus=18 --num-dirs=18 --topology=Mesh_XY --mesh-rows=6 \
        --sim-cycles=50000000 --injectionrate=0 --network-trace-enable \
        --network-trace-file="1_757_files/PreActResNet18/PreActResNet18_bits_"$bits"_garnet_traffic.txt.gz" \
        --network-trace-max-packets=$max_pkts 

sleep 1


echo "power run"
python util/on-chip-network-power-area.py .  \
        m5out \
        ext/dsent/configs/garnet_router.cfg \
        ext/dsent/configs/garnet_link.cfg \
        32 1000 > log_power