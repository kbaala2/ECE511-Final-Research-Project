import numpy as np

# standard_chiplets = [1,2,3,6,7,8,9,10,11,15], 4 nodes, 4.55 w power
# adder_chiplets = [16,17,18,20,21,22], 4 nodes, 31.47 w power
# accumulator_chiplets = [4,5,12,13], 9 nodes, 0.91 w power
# adcless_chiplets = [14,19], 9 nodes, 57.25 w power


# dictionary of chiplet-node placement
# chiplet_node_placement = {  1 : [174, 4, 4.55],
#                             2:  [178, 4, 4.55],
#                             3:  [182, 4, 4.55],
#                             4:  [186, 9, 0.91],
#                             5:  [195, 9, 0.91],
#                             6:  [204, 4, 4.55],
#                             7:  [208, 4, 4.55],
#                             8:  [212, 4, 4.55],
#                             9:  [216, 4, 4.55],
#                             10: [220, 4, 4.55],
#                             11: [224, 4, 4.55],
#                             12: [228, 9, 0.91],
#                             13: [237, 9, 0.91],
#                             14: [246, 9, 57.25],
#                             15: [255, 4, 4.55],
#                             16: [259, 4, 31.47],
#                             17: [263, 4, 31.47],
#                             18: [267, 4, 31.47],
#                             19: [271, 9, 57.25],
#                             20: [280, 4, 31.47],
#                             21: [284, 4, 31.47],
#                             22: [288, 4, 31.47]}

chiplet_node_placement = {  1 : [174, 4, 0.5],
                            2:  [178, 4, 0.5],
                            3:  [182, 4, 0.5],
                            4:  [186, 9, 3],
                            5:  [195, 9, 3],
                            6:  [204, 4, 0.5],
                            7:  [208, 4, 0.5],
                            8:  [212, 4, 0.5],
                            9:  [216, 4, 0.5],
                            10: [220, 4, 0.5],
                            11: [224, 4, 0.5],
                            12: [228, 9, 3],
                            13: [237, 9, 3],
                            14: [246, 9, 0.5],
                            15: [255, 4, 0.5],
                            16: [259, 4, 0.5],
                            17: [263, 4, 0.5],
                            18: [267, 4, 0.5],
                            19: [271, 9, 0.5],
                            20: [280, 4, 0.5],
                            21: [284, 4, 0.5],
                            22: [288, 4, 0.5]}

chiplet_clusters = { 0 : np.array([0,1,2,5,6,7,8,9,10,14]), # standard chiplets
                     1 : np.array([3,4,11,12]), # accumulator chiplets
                     2 : np.array([13,18]), # adcless chiplets
                     3 : np.array([15,16,17,19,20,21])} # adder chiplets

class DSS:
    def __init__(self, disc_A, disc_B, chiplet_nodes=chiplet_node_placement, ts=0.0001, tp=1):
        self.disc_A = disc_A
        self.disc_B = disc_B
        self.ts = ts #time step of thermal sampling, do not change this as its fixed for given disc_A and disc_B
        self.tp = tp #time step of power sampling, do not change this as its fixed for given disc_A and disc_B
        self.iteration = int(tp/ts) #tp/ts

        self.io_power = 2 #fixed 10w power for io nodes
        self.io_nodes = {23: [168, 3],
                         24: [171, 3],
                         25: [292, 3],
                         26: [295, 3]}
        
        self.chiplet_nodes = chiplet_nodes

    def run_dss(self, temperature_initial, power_sequence):
        '''
        power_sequence: list of power values for each chiplet (22)
        temperature_initial: initial temperature of each node (size(disc_A))
        '''

        power_sequence_node = self.generate_power_seq_node(power_sequence)

        T_predicted = np.zeros((temperature_initial.shape[0], self.iteration+1))
        T_predicted[:, 0] = temperature_initial

        for i in range(self.iteration):
            T_predicted[:, i+1] = np.dot(self.disc_A, T_predicted[:, i]) + np.dot(self.disc_B, power_sequence_node)

        chiplet_temperature = self.map_temperature_to_chiplet(T_predicted)

        return T_predicted[:, -1], chiplet_temperature
    
    def generate_power_seq_node(self, power_sequence):
        power_sequence_node = np.zeros(self.disc_A.shape[0])

        for io_chiplet, nodes in self.io_nodes.items():
            power_sequence_node[nodes[0]: nodes[0]+ nodes[1]] = self.io_power/ nodes[1]

        for chiplet, nodes in self.chiplet_nodes.items():
            power_sequence_node[nodes[0]: nodes[0]+ nodes[1]] = power_sequence[chiplet-1]/(nodes[1]) # power in watts
            # power_sequence_node[nodes[0]: nodes[0]+ nodes[1]] = (power_sequence[chiplet-1]*nodes[2])/ (nodes[1]*100) # power in percentage

        return power_sequence_node
    
    def map_temperature_to_chiplet(self, temperature):
        # chiplet_temperature is length of chiplet_nodes dictionary
        chiplet_temperature = np.zeros(len(self.chiplet_nodes))
        for chiplet, nodes in self.chiplet_nodes.items():
            chiplet_temperature[chiplet-1] = np.mean(temperature[nodes[0]: nodes[0]+ nodes[1], -1])
        
        return chiplet_temperature + 300.0

# main function
if __name__ == '__main__':

    disc_A_small_cluster = np.loadtxt(f'disc_A_matrix.csv', delimiter=',')
    disc_B_small_cluster = np.loadtxt(f'disc_B_matrix.csv', delimiter=',')

    dss_model = DSS(disc_A=disc_A_small_cluster, disc_B=disc_B_small_cluster)

    temperature_initial = np.zeros(disc_A_small_cluster.shape[0])

    power_sequence = np.zeros(22) + 50

    for i in range(6):
        temperature_initial, chiplet_temperature = dss_model.run_dss(temperature_initial, power_sequence)
        print(f'ms {i+1} done')
        print(f'Chiplet temperature: {chiplet_temperature}')
        print('-------------------------------------------')
