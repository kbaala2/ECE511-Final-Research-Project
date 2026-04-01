"""
Stats Collector Module

Generic utility for collecting and persisting simulation statistics to prevent memory overflow.
Supports batched writes to JSONL format for efficient streaming analysis.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any


class StatsCollector:
    """
    Generic collector for simulation statistics with batched file persistence.
    
    This class manages a buffer of statistics entries and periodically dumps them
    to a JSONL file to prevent memory overflow during long simulations.
    
    Args:
        stats_file_path (str): Path to the output JSONL file
        dump_threshold (int): Number of entries to buffer before auto-dumping
        stats_type (str): Type of statistics being collected (e.g., "dsent", "timing")
        auto_initialize_file (bool): Whether to clear/initialize the file on creation
    """
    
    def __init__(self, 
                 stats_file_path: str, 
                 dump_threshold: int = 1,
                 stats_type: str = "generic",
                 auto_initialize_file: bool = True,
                 serialization_format: str = "json"):
        self.stats_file_path = stats_file_path
        self.dump_threshold = dump_threshold
        self.stats_type = stats_type
        self.stats_buffer: List[Dict[str, Any]] = []
        self.serialization_format = serialization_format
        self._logger = None  # Optional per-type logger
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
        
        # Initialize/clear file if requested
        if auto_initialize_file:
            self._initialize_file()

        # DEBUG DSENT: set up a dedicated logger for DSENT stats in temp/logs (easy to remove later)
        if self.stats_type == "dsent":
            try:
                logs_dir = os.path.join(os.getcwd(), "temp", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                logger = logging.getLogger("dsent_debug")
                if not logger.handlers:
                    logger.setLevel(logging.INFO)
                    fh = logging.FileHandler(os.path.join(logs_dir, "dsent_debug.log"))
                    fh.setLevel(logging.INFO)
                    fmt = logging.Formatter("%(asctime)s | %(message)s")
                    fh.setFormatter(fmt)
                    logger.addHandler(fh)
                self._logger = logger
            except Exception:
                # If logger setup fails, we silently skip logging
                self._logger = None
    
    def _initialize_file(self):
        """Clear/initialize the stats file at the start"""
        try:
            mode = 'wb' if self.serialization_format == "pickle" else 'w'
            with open(self.stats_file_path, mode) as f:
                pass  # Just clear the file
        except Exception as e:
            print(f"⚠️ WARNING: Could not initialize stats file {self.stats_file_path}: {e}")
    
    def add_stats(self, stats_entry: Dict[str, Any]):
        """
        Add a statistics entry to the buffer.
        Automatically dumps to file if threshold is reached.
        
        Args:
            stats_entry (dict): Statistics entry to add
        """
        self.stats_buffer.append(stats_entry)

        # DEBUG DSENT: track when DSENT stats are added (logged to temp/logs/dsent_debug.log)
        if self.stats_type == "dsent" and self._logger:
            try:
                self._logger.info(
                    f"add_stats: buffered {len(self.stats_buffer)} entries "
                    f"(dump_threshold={self.dump_threshold}) -> file={self.stats_file_path}"
                )
            except Exception:
                pass
        
        # Auto-dump if threshold reached
        if len(self.stats_buffer) >= self.dump_threshold:
            if self.stats_type == "dsent" and self._logger:
                try:
                    self._logger.info(
                        f"add_stats: triggering dump (buffer={len(self.stats_buffer)}, "
                        f"threshold={self.dump_threshold})"
                    )
                except Exception:
                    pass
            self.dump_stats()
    
    def dump_stats(self):
        """
        Dump the collected stats to file and clear the buffer.
        Appends to the JSONL file (one JSON object per line).
        """
        if not self.stats_buffer:
            return
        
        try:
            if self.serialization_format == "pickle":
                with open(self.stats_file_path, 'ab') as f:
                    for entry in self.stats_buffer:
                        pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.stats_file_path, 'a') as f:
                    for entry in self.stats_buffer:
                        f.write(json.dumps(entry) + '\n')
            
            # DEBUG DSENT: log dumps for DSENT stats specifically (to temp/logs/dsent_debug.log)
            if self.stats_type == "dsent" and self._logger:
                try:
                    self._logger.info(
                        f"dump_stats: wrote {len(self.stats_buffer)} entries to {self.stats_file_path}"
                    )
                except Exception:
                    pass
            elif len(self.stats_buffer) >= 10:  # Existing behavior for larger dumps of other types
                print(f"INFO: Dumped {len(self.stats_buffer)} {self.stats_type} stats to {self.stats_file_path}")
            
            # Clear the buffer after successful dump
            self.stats_buffer.clear()
            
        except Exception as e:
            if self.stats_type == "dsent" and self._logger:
                try:
                    self._logger.error(
                        f"dump_stats failed for file={self.stats_file_path}: {e}",
                        exc_info=True
                    )
                except Exception:
                    pass
            print(f"❌ ERROR: Could not dump {self.stats_type} stats to file: {e}")
    
    def finalize(self):
        """
        Dump any remaining stats in the buffer.
        Should be called at the end of simulation.
        """
        self.dump_stats()
    
    def get_buffer_size(self) -> int:
        """
        Get the current number of entries in the buffer.
        
        Returns:
            int: Number of buffered entries
        """
        return len(self.stats_buffer)
    
    def clear_buffer(self):
        """Clear the buffer without dumping (use with caution)"""
        self.stats_buffer.clear()

