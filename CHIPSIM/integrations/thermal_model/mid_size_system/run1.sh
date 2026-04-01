
# This is most basic configuration where all the chiplets are identical. chiplet_geometry_4_chiplets_uniform_nodes.yml that nodes are defined as number of nodes in x and y direction for each layer.

cd ../
python thermal_RC.py --material_prop_file material_prop.yml \
                       --geometry_file mid_size_system/geometry_mid_size.yml \
                       --power_config_file mid_size_system/power_dist_mid_size.yml \
                       --power_seq_file mid_size_system/power_seq_mid_size.csv \
                       --output_dir mid_size_system/
cd -
# Other parameters are set to default values.
# From output floorplan or heatmaps notice that the nodes are uniformly distributed in each layer. Still the nodes can have different granularity by changing `x_nodes`, `y_nodes` in the geometry file.
