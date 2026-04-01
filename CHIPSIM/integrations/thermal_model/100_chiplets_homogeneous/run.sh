# This is most basic configuration where all the chiplets are identical. chiplet_geometry_4_chiplets_uniform_nodes.yml that nodes are defined as number of nodes in x and y direction for each layer.

cd ../
python3 thermal_RC.py --material_prop_file material_prop.yml \
                       --geometry_file 100_chiplets_homogeneous/system_geometry.yaml \
                       --power_config_file 100_chiplets_homogeneous/power_dist.yaml \
                       --power_seq_file 100_chiplets_homogeneous/chiplet_power_1ms_avg.csv \
                       --output_dir 100_chiplets_homogeneous/ \
                       --is_homogeneous True \
                       --generate_DSS False \
                       --generate_heatmap True \
                       --time_step 0.001 \
                       --simulation_type transient \
                       --power_interval 0.001  \
                       --total_duration 0.01 \
                       --time_heatmap 0.0 \

cd -
# Other parameters are set to default values.
# From output floorplan or heatmaps notice that the nodes are uniformly distributed in each layer. Still the nodes can have different granularity by changing `x_nodes`, `y_nodes` in the geometry file.
