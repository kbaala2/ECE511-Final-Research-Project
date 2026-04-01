"""
Analytical Backend for Compute Simulation

Handles compute simulation for CMOS chiplets using an analytical model.
Calculates latency and energy based on total MACs and chiplet parameters.
"""


class AnalyticalBackend:
    """
    Backend for simulating CMOS chiplets using analytical models.
    """
    
    def __init__(self):
        """Initialize the Analytical backend."""
        pass
    
    def simulate(self, partitioned_layer, chiplet_id, chiplet_params, batch_size=1):
        """
        Simulate compute using analytical model.
        
        Args:
            partitioned_layer (dict): Partitioned layer definition with 'total_macs' field
            chiplet_id (int): ID of the chiplet to simulate on
            chiplet_params (dict): Chiplet parameters including 'macs_per_second' and 'energy_per_mac'
            batch_size (int): Batch size for simulation
            
        Returns:
            dict: Simulation results containing latency_us, energy_fj, cycles
            
        Raises:
            ValueError: If required parameters are missing
        """
        print(f"🔄 Running analytical simulation for chiplet {chiplet_id}...")
        
        # Validate required parameters
        if 'total_macs' not in partitioned_layer:
            raise ValueError(f"Missing 'total_macs' field in partitioned layer for chiplet {chiplet_id}")
        
        if 'macs_per_second' not in chiplet_params:
            raise ValueError(f"Missing 'macs_per_second' parameter for CMOS chiplet {chiplet_id}")
        
        if 'energy_per_mac' not in chiplet_params:
            raise ValueError(f"Missing 'energy_per_mac' parameter for CMOS chiplet {chiplet_id}")
        
        # Extract parameters
        total_macs = partitioned_layer['total_macs']
        macs_per_second = chiplet_params['macs_per_second']
        energy_per_mac = chiplet_params['energy_per_mac']
        
        # Calculate latency (in microseconds)
        # latency_seconds = total_macs / macs_per_second
        # latency_us = latency_seconds * 1e6
        latency_us = (total_macs / macs_per_second) * 1e6
        
        # Calculate energy (in femtojoules)
        energy_fj = total_macs * energy_per_mac
        
        # Set cycles to -1 (not calculated for analytical model)
        cycles = -1
        
        # Log the results
        print(f"✅ Analytical simulation results: Latency={latency_us:,.2f} μs, Energy={energy_fj:,.2f} fJ, MACs={total_macs:,.0f}")
        
        # Create the result dictionary
        result = {
            'latency_us': latency_us,
            'energy_fj': energy_fj,
            'cycles': cycles
        }
        
        return result

