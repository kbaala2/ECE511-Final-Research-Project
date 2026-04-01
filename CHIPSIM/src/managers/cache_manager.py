import os
import json
import numpy as np
import hashlib
import pickle

class CacheManager:
    """
    Handles caching of simulation results to avoid redundant computations.
    """
    def __init__(self, cache_file, clear_cache=False):
        """
        Initialize the cache manager.
        
        Args:
            cache_file (str): Path to the cache file
            clear_cache (bool): Whether to clear the existing cache
        """
        self.cache_file = cache_file
        self.clear_cache = clear_cache
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """
        Load the cache from file or create an empty cache.
        
        Returns:
            dict: The cache data
        """
        # If clear_cache is True, delete existing cache file and return empty cache
        if self.clear_cache:
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to clear cache at {self.cache_file}: {e}")
            return {}

        # Otherwise load the cache if it exists
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read/parse cache file {self.cache_file}: {e}")
            # Cache stored as native Python structures (including tuple keys)
            return loaded_cache if isinstance(loaded_cache, dict) else {}
        else:
            # Create the directory if it doesn't exist
            try:
                directory = os.path.dirname(self.cache_file)
                if directory:
                    os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create cache directory for {self.cache_file}: {e}")
            # Return an empty cache
            return {}
    
    def save_cache(self):
        """
        Save the cache to file.
        """
        # Atomic write (pickle) to avoid partial/corrupt files
        tmp_path = f"{self.cache_file}.tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.cache_file)
        except Exception as e:
            # Cleanup tmp file if present
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"Failed to write cache to {self.cache_file}: {e}")
    
    # Legacy JSON conversion helpers removed; cache is stored as native Python via pickle

    def compute_cache_key(self, **params):
        """
        Compute a unique hash key for the provided parameters.
        
        Args:
            **params: Variable keyword arguments to include in the key
            
        Returns:
            str: Hash key for the parameters
        """
        # Create a copy of params that we can modify
        serializable_params = {}
        
        # Convert numpy arrays to lists for serialization
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            else:
                serializable_params[key] = value
        
        # Create a string with all parameters
        param_str = json.dumps(serializable_params, sort_keys=True)
        
        # Compute the SHA-256 hash
        hash_obj = hashlib.sha256(param_str.encode())
        return hash_obj.hexdigest()
    
    def get_result(self, key):
        """
        Get a result from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            object: Cached result or None if not found
        """
        if key not in self.cache:
            return None
        value = self.cache.get(key)
        # Treat empty or None cached entries as errors to avoid infinite loops upstream
        if value is None:
            raise RuntimeError(f"Cache error: key {key} exists but value is None")
        if isinstance(value, dict) and len(value) == 0:
            raise RuntimeError(f"Cache error: key {key} has empty dict value")
        if isinstance(value, (list, tuple)) and len(value) == 0:
            raise RuntimeError(f"Cache error: key {key} has empty sequence value")
        return value
    
    def has_result(self, key):
        """
        Check if a result exists in the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if the key exists in the cache
        """
        return key in self.cache
    
    def store_result(self, key, result):
        """
        Store a result in the cache.
        
        Args:
            key (str): Cache key
            result (object): Result to store
        """
        if result is None:
            raise RuntimeError(f"Refusing to store None for cache key {key}")
        self.cache[key] = result
        self.save_cache() 