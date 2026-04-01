"""
Performance Monitor Module

Generic utility for tracking and reporting timing statistics for simulation operations.
Provides context managers and decorators for easy timing instrumentation.
"""

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Any, Optional


class PerformanceMonitor:
    """
    Tracks timing statistics for simulation operations and generates performance reports.
    
    Usage:
        monitor = PerformanceMonitor(log_file_path="timing.log")
        
        # Context manager approach
        with monitor.time_operation('operation_name'):
            # Your operation here
            pass
        
        # Decorator approach
        @monitor.time_method('method_name')
        def my_method(self):
            # Your method here
            pass
    
    Args:
        log_file_path (str, optional): Path to write timing summary logs
        auto_initialize_log (bool): Whether to initialize the log file on creation
    """
    
    def __init__(self, log_file_path: Optional[str] = None, auto_initialize_log: bool = True):
        self.timing_stats: Dict[str, Dict[str, float]] = {}
        self.log_file_path = log_file_path
        
        # Initialize log file if path provided
        if self.log_file_path and auto_initialize_log:
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the timing log file"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            
            # Clear log file at the start
            with open(self.log_file_path, 'w') as f:
                f.write("Timing Summary Log Initialized\n")
                f.write("=" * 30 + "\n")
        except Exception as e:
            print(f"⚠️ WARNING: Could not initialize timing log file: {e}")
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations and automatically updating timing stats.
        
        Usage:
            with monitor.time_operation('operation_name'):
                # Your operation code here
                pass
        
        Args:
            operation_name (str): Name to identify this operation in statistics
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_operation(operation_name, duration)
    
    def time_method(self, operation_name: str):
        """
        Decorator for timing methods and automatically updating timing stats.
        
        Usage:
            @monitor.time_method('operation_name')
            def some_method(self):
                # Your method code here
                pass
        
        Args:
            operation_name (str): Name to identify this operation in statistics
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_operation(operation_name, duration)
            return wrapper
        return decorator
    
    def record_operation(self, operation_name: str, duration: float):
        """
        Record timing data for an operation.
        
        Args:
            operation_name (str): Name of the operation
            duration (float): Duration in seconds
        """
        if operation_name not in self.timing_stats:
            self.timing_stats[operation_name] = {'total_time': 0.0, 'count': 0}
        
        self.timing_stats[operation_name]['total_time'] += duration
        self.timing_stats[operation_name]['count'] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current timing statistics.
        
        Returns:
            dict: Dictionary mapping operation names to their timing stats
        """
        return self.timing_stats.copy()
    
    def get_average_time(self, operation_name: str) -> float:
        """
        Get the average time for a specific operation.
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            float: Average time in seconds, or 0 if operation not found
        """
        if operation_name in self.timing_stats:
            stats = self.timing_stats[operation_name]
            if stats['count'] > 0:
                return stats['total_time'] / stats['count']
        return 0.0
    
    def log_timing_summary(self, simulation_duration: float):
        """
        Write timing summary to the log file.
        
        Args:
            simulation_duration (float): Total simulation duration in seconds
        """
        if not self.log_file_path:
            print("⚠️ WARNING: No log file path configured for PerformanceMonitor")
            return
        
        try:
            summary_header = "\nTiming Summary Report\n" + "-" * 30
            
            print(summary_header)
            
            with open(self.log_file_path, 'a') as f:
                f.write(f"Overall Simulation Duration: {simulation_duration:.6f} seconds\n")
                f.write(summary_header + "\n")
                f.write(f"{'Section':<35} | {'Total Time (s)':<15} | {'Count':<10} | {'Avg Time (s)':<15}\n")
                f.write("-" * 80 + "\n")
                
                for section, stats in self.timing_stats.items():
                    total_time = stats['total_time']
                    count = stats['count']
                    avg_time = total_time / count if count > 0 else 0
                    summary_line = f"{section:<35} | {total_time:<15.6f} | {count:<10} | {avg_time:<15.6f}"
                    print(summary_line)
                    f.write(summary_line + "\n")
                
                f.write("=" * 80 + "\n")
            
            print("Timing summary logged to", self.log_file_path)
            
        except Exception as e:
            print(f"❌ ERROR: Could not write timing summary: {e}")
    
    def print_summary(self):
        """Print a summary of timing statistics to console"""
        print("\n" + "=" * 80)
        print("Performance Monitor Summary")
        print("=" * 80)
        print(f"{'Operation':<35} | {'Total (s)':<12} | {'Count':<8} | {'Avg (s)':<12}")
        print("-" * 80)
        
        for operation_name, stats in sorted(self.timing_stats.items()):
            total_time = stats['total_time']
            count = stats['count']
            avg_time = total_time / count if count > 0 else 0
            print(f"{operation_name:<35} | {total_time:<12.4f} | {count:<8} | {avg_time:<12.6f}")
        
        print("=" * 80)
    
    def reset(self):
        """Reset all timing statistics"""
        self.timing_stats.clear()

