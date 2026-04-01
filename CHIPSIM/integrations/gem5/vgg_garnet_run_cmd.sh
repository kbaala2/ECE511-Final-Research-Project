#!/bin/bash

# --debug-flags=NetworkTrace  \

bits=24

case $bits in
        0)
        max_pkts=7439
        ;;
        4)
        max_pkts=6509
        ;;
        8)
        max_pkts=5579
        ;;
        12)
        max_pkts=4649
        ;;
        16)
        max_pkts=3719
        ;;
        20)
        max_pkts=2789
        ;;
        22)
        max_pkts=2324
        ;;
        24)
        max_pkts=1859
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
        --network-trace-file="1_757_files/VGG19/VGG19_bits_"$bits"_garnet_traffic.txt.gz" \
        --network-trace-max-packets=$max_pkts 

sleep 1


echo "power run"
python util/on-chip-network-power-area.py .  \
        m5out \
        ext/dsent/configs/garnet_router.cfg \
        ext/dsent/configs/garnet_link.cfg \
        32 1000 > log_power