import numpy as np

# standard_chiplets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], 4 nodes, 1W power
# accumulator_chiplets = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], 2 nodes, 1W power
# adcless_chiplets = [41,42,43,44,45,46,47,48,49,50], 2 nodes, 1W power
# sharedadc_chiplets = [51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78], 4 nodes, 1W power

chiplet_node_placement = {  1 : [146, 4, 1],
                            2 : [186, 4, 1],
                            3 : [226, 4, 1],
                            4 : [266, 4, 1],
                            5 : [306, 4, 1],
                            6 : [150, 4, 1],
                            7 : [190, 4, 1],
                            8 : [230, 4, 1],
                            9 : [270, 4, 1],
                            10 : [310, 4, 1],
                            11 : [154, 4, 1],
                            12 : [194, 4, 1],
                            13 : [234, 4, 1],
                            14 : [274, 4, 1],
                            15 : [314, 4, 1],
                            16 : [158, 4, 1],
                            17 : [198, 4, 1],
                            18 : [238, 4, 1],
                            19 : [278, 4, 1],
                            20 : [318, 4, 1],
                            21 : [162, 4, 1],
                            22 : [202, 4, 1],
                            23 : [242, 4, 1],
                            24 : [282, 4, 1],
                            25 : [322, 4, 1],
                            26 : [166, 4, 1],
                            27 : [206, 4, 1],
                            28 : [246, 4, 1],
                            29 : [286, 4, 1],
                            30 : [326, 4, 1],
                            31 : [170, 4, 1],
                            32 : [210, 4, 1],
                            33 : [250, 4, 1],
                            34 : [290, 4, 1],
                            35 : [330, 4, 1],
                            36 : [174, 4, 1],
                            37 : [214, 4, 1],
                            38 : [254, 4, 1],
                            39 : [294, 4, 1],
                            40 : [334, 4, 1],
                            41 : [178, 4, 1],
                            42 : [218, 4, 1],
                            43 : [258, 4, 1],
                            44 : [298, 4, 1],
                            45 : [338, 4, 1],
                            46 : [182, 4, 1],
                            47 : [222, 4, 1],
                            48 : [262, 4, 1],
                            49 : [302, 4, 1],
                            50 : [342, 4, 1],
                            51 : [346, 4, 1],
                            52 : [374, 4, 1],
                            53 : [402, 4, 1],
                            54 : [430, 4, 1],
                            55 : [350, 4, 1],
                            56 : [378, 4, 1],
                            57 : [406, 4, 1],
                            58 : [434, 4, 1],
                            59 : [354, 4, 1],
                            60 : [382, 4, 1],
                            61 : [410, 4, 1],
                            62 : [438, 4, 1],
                            63 : [358, 4, 1],
                            64 : [386, 4, 1],
                            65 : [414, 4, 1],
                            66 : [442, 4, 1],
                            67 : [362, 4, 1],
                            68 : [390, 4, 1],
                            69 : [418, 4, 1],
                            70 : [446, 4, 1],
                            71 : [366, 4, 1],
                            72 : [394, 4, 1],
                            73 : [422, 4, 1],
                            74 : [450, 4, 1],
                            75 : [370, 4, 1],
                            76 : [398, 4, 1],
                            77 : [426, 4, 1],
                            78 : [454, 4, 1]}

chiplet_clusters = { 0 : np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]), # standard chiplets
                     1 : np.array([26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]), # accumulator chiplets
                     2 : np.array([41,42,43,44,45,46,47,48,49,50]), # adcless chiplets
                     3 : np.array([51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78])} # sharedadc chiplets

io_nodes = {79: [134, 3],
            80: [137, 3],
            81: [140, 3],
            82: [143, 3],
            83: [458, 3],
            84: [461, 3],
            85: [464, 3],
            86: [467, 3]}

class DSS:
    def __init__(self, disc_A, disc_B, chiplet_nodes=chiplet_node_placement, ts=0.1, tp=1, io_nodes=io_nodes):
        self.disc_A = disc_A
        self.disc_B = disc_B
        self.ts = ts #time step of thermal sampling, do not change this as its fixed for given disc_A and disc_B
        self.tp = tp #time step of power sampling, do not change this as its fixed for given disc_A and disc_B
        self.iteration = int(tp/ts) #tp/ts

        self.io_power = 2.5 #fixed 10w power for io nodes
        self.io_nodes = io_nodes
        
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

    power_sequence = np.zeros(78) + 50

    for i in range(6):
        temperature_initial, chiplet_temperature = dss_model.run_dss(temperature_initial, power_sequence)
        print(f'ms {i+1} done')
        print(f'Chiplet temperature: {chiplet_temperature}')
        print('-------------------------------------------')
