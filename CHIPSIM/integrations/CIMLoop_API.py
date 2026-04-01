import requests
import time

class CIMLoop_API:
    def __init__(self, base_url="http://localhost:5000"):
        """
        Initialize the CimLoop API client.
        
        Args:
            base_url (str): The base URL of the CimLoop API server.
        """
        self.base_url = base_url
        self.simulation_url = f"{self.base_url}/run_simulation_with_yaml"
        self.health_url = f"{self.base_url}/health"

    def run_simulation_with_yaml(self, yaml_configs=None, batch_size=1, system_config=None, chiplet_type=None, debug=False):
        """
        Run a simulation using the CimLoop API with YAML configurations.
        
        Args:
            yaml_configs (list): List of YAML configuration strings.
            batch_size (int): Batch size for the simulation.
            system_config (str): System configuration name.
            chiplet_type (str): Type of the chiplet being simulated.
            debug (bool): Whether to print additional debugging information.
            
        Returns:
            dict or None: Simulation results, or None if the API call failed.
        """
        # Default YAML config if none provided
        if yaml_configs is None:
            # Throw an error and exit
            raise ValueError("No YAML configurations provided")
        
        payload = {
            "yaml_configs": yaml_configs,
            "batch_size": batch_size,
            "system_config": system_config,
            "chiplet_type": chiplet_type
        }
        
        if debug:
            print("\n=== API REQUEST DETAILS ===")
            print(f"URL: {self.simulation_url}")
            print(f"Batch Size: {batch_size}")
            print(f"System Config: {system_config}")
            print(f"Chiplet Type: {chiplet_type}")
            print(f"Number of YAML Configs: {len(yaml_configs)}")
            
            if len(yaml_configs[0]) < 500:
                print("\nFirst YAML Config:")
                print("-" * 40)
                print(yaml_configs[0])
                print("-" * 40)
            else:
                print(f"\nFirst YAML Config (showing first 500 chars of {len(yaml_configs[0])} total):")
                print("-" * 40)
                print(yaml_configs[0][:500] + "...")
                print("-" * 40)
            
        try:
            start_time = time.time()
            
            # Send the actual request
            response = requests.post(self.simulation_url, json=payload, timeout=60)  # 60 second timeout
            api_time = time.time() - start_time
            
            # Check if we got a 500 error
            if response.status_code == 500:
                print(f"Error: CimLoop API server returned 500 Internal Server Error")
                print("This usually indicates a problem with the YAML configuration or the server itself.")
                print("\nResponse details:")
                try:
                    error_detail = response.json()
                    print(f"  Server Error Message: {error_detail.get('message', 'No message available')}")
                    if 'traceback' in error_detail:
                        print("\n--- Server Traceback ---")
                        print(error_detail['traceback'])
                        print("------------------------")
                except:
                    print(f"  Raw response: {response.text[:200]}...")
                
                return None
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            if debug:
                print(f"\n=== YAML-based Simulation Results ===")
                print(f"Time Taken: {api_time:.4f} seconds")
                print(f"Status: {result.get('status', 'N/A')}")
                
                # Handle runtime (in seconds, convert to ms for display)
                runtime_seconds = result.get('total_runtime_seconds')
                if runtime_seconds is not None:
                    print(f"Total Runtime: {runtime_seconds * 1000:.4f} ms")
                else:
                    print("Total Runtime: Not available")
                    
                # Handle energy (already in fJ)
                total_energy = result.get('total_energy_fJ')
                if total_energy is not None:
                    print(f"Total Energy: {total_energy/1e6:.4f} nJ ({total_energy:.2f} fJ)")
                else:
                    print("Total Energy: Not available")
                    
                # Print cycles if available
                total_cycles = result.get('total_cycles')
                if total_cycles is not None:
                    print(f"Total Cycles: {total_cycles:,}")
                    
                # Check if layer breakdown is available
                layer_breakdown = result.get('layer_breakdown')
                if layer_breakdown:
                    print("\nLayer-by-layer energy breakdown:")
                    for layer in layer_breakdown:
                        print(f"  Layer {layer['layer']}: {layer['energy_fJ']:.2f} fJ")
            
            return result
        
        except requests.exceptions.Timeout:
            print(f"Error: Connection to CimLoop API timed out after {time.time() - start_time:.1f} seconds")
            print("The server may be busy processing another request or not responding.")
            return None
            
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to CimLoop API at {self.simulation_url}")
            print("Ensure the API server is running and accessible.")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to YAML Simulation API: {e}")
            
            # More detailed error for common HTTP errors
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                
                if status_code == 400:
                    print("This is a Bad Request error. The YAML configuration might be invalid.")
                elif status_code == 404:
                    print("The API endpoint was not found. Check the URL and ensure the API server is running.")
                elif status_code == 500:
                    print("The server encountered an internal error. Check the server logs for details.")
            
            return None

    def test_api_connection(self):
        """
        Test the connection to the CimLoop API server.
        
        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        # First try the health endpoint
        try:
            # Try to connect to the health endpoint
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                print("✅ CimLoop API server is running and has a healthy endpoint.")
                return True
            elif response.status_code == 404:
                print("ℹ️ CimLoop API server is running but doesn't have a /health endpoint.")
                print("This is okay - checking actual API endpoint...")
                
                # If health endpoint doesn't exist, check if the main endpoint exists
                try:
                    # Make a HEAD request to the main endpoint (doesn't actually send data)
                    response = requests.head(self.simulation_url, timeout=5)
                    
                    # 405 Method Not Allowed is actually good - it means the endpoint exists
                    # but doesn't support HEAD requests (typically only POST)
                    if response.status_code in [200, 405, 400]:
                        print("✅ CimLoop API endpoint exists and is accessible.")
                        return True
                    else:
                        print(f"❌ CimLoop API endpoint returned unexpected status: {response.status_code}")
                        return False
                except requests.exceptions.RequestException as e:
                    print(f"❌ Failed to connect to main CimLoop API endpoint: {e}")
                    return False
            else:
                print(f"⚠️ CimLoop API server health check returned unexpected status code: {response.status_code}")
                
                # Try the main endpoint as a fallback
                try:
                    response = requests.head(self.simulation_url, timeout=5)
                    if response.status_code in [200, 405, 400]:
                        print("✅ Main API endpoint exists despite health check issues.")
                        return True
                    else:
                        return False
                except:
                    return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to CimLoop API server: {e}")
            print(f"Ensure the server is running at {self.base_url}")
            
            # Try a more basic connection check as a last resort
            try:
                requests.get(self.base_url, timeout=2)
                print("ℹ️ The server is responding at the base URL, but specific endpoints may be different.")
                print("You might need to check the API documentation for the correct endpoints.")
                return False
            except:
                print(f"❌ Could not establish any connection to {self.base_url}")
                return False

if __name__ == '__main__':
    # Test the API connection
    api = CIMLoop_API()
    api.test_api_connection()
