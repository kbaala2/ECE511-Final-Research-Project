# Chiplet performance parameters (for different chiplet types)
CHIPLET_TYPES = [
    {
        'name': 'IMC_A',            # Standard or Basic Analog
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 873.109504,      # in fJ
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 2,         # Resolution of the individual crossbarl cells
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 299,
    },
    {
        'name': 'IMC_B',           # Raella
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 873.109504,      # in fJ
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 2,         # Resolution of the individual crossbarl cells
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 299,
    },
    { 
        'name': 'IMC_C',           # Macro A, jia_jssc_2020
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 303.928319999999,
        'crossbar_rows': 768,
        'crossbar_columns': 768,
        'bits_per_cell': 1,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 17,
    },
    {
        'name': 'IMC_D',           # Macro B, sinangil_jssc_2021
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 177.2664832,     
        'crossbar_rows': 64,
        'crossbar_columns': 64,
        'bits_per_cell': 1,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 216,
    },
    {
        'name': 'IMC_E',            # Macro C, wang_vlsi_2022
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 222.45888,
        'crossbar_rows': 256,
        'crossbar_columns': 256,
        'bits_per_cell': 2,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 150,
    },
    {
        'name': 'IMC_F',            # Macro E, colonnade jssc 2021
        'type': 'IMC',              # In-Memory Computing chiplet
        'energy_per_mac': 268.704,
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 1,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 151,
    },
    {
        'name': 'IO',               # I/O chiplet for weight loading
        'type': 'IO',               # I/O chiplet type
        'energy_per_mac': 0,        # No MAC operations
        'crossbar_rows': 0,         # No crossbars
        'crossbar_columns': 0,      # No crossbars
        'bits_per_cell': 0,         # No memory cells
        'bits_per_weight': 8,       # Weights come from external memory (default precision)
        'tiles_per_chiplet': 0,     # No tiles
        'crossbars_per_tile': 0,    # No crossbars
    },
    {
        'name': 'CMOS_Compute',
        'type': 'CMOS',            # CMOS chiplet type
        'macs_per_second': 1e12,   # 1 TMAC/s (1 TeraMAC per second)
        'energy_per_mac': 50.0,    # in fJ
        'crossbar_rows': 0,        # CMOS chiplets don't use crossbars
        'crossbar_columns': 0,
        'bits_per_cell': 0,
        'dac_precision_bits': 0,
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 0,
        'crossbars_per_tile': 0,
        # Total memory (weights) for CMOS is defined to match Accumulator capacity
        # Accumulator: rows=256, cols=256, bits_per_cell=2, bits_per_weight=8, crossbars=150
        # memory_per_crossbar = 256*256*2/8 = 16384; total = 16384 * 150 = 2,457,600
        'total_memory_weights': 2457600,
    },
]
