#!/usr/bin/env python3
# simulate.py
"""
Chiplet System Simulator - Main Entry Point

This script provides a unified interface for running simulations, reprocessing results,
and performing cross-simulation analysis.

Usage:
    # Run a simulation (with automatic post-processing)
    python3 simulate.py --mode simulate --config config_1
    
    # Re-process existing results
    python3 simulate.py --mode reprocess --config config_1 --results-dir "2025.10.07_12.30.45_..."
    
    # Cross-simulation analysis
    python3 simulate.py --mode cross-analysis --config compare_pipelines
    
    # Batch simulations
    python3 simulate.py --mode batch --configs config_1,config_2,config_3
"""

import argparse
import os
import sys

from src.run import SimulationRunner
from src.post.single_sim_processor import SingleSimProcessor
from src.post.cross_sim_processor import CrossSimProcessor
from src.utils.config_loader import load_config


# ============================================
# CONFIGURABLE DEFAULTS (Edit these directly)
# ============================================
DEFAULT_MODE = "simulate"  # Options: "simulate", "reprocess", "cross-analysis", "batch"
DEFAULT_CONFIG = "config_1"
DEFAULT_RESULTS_DIR = None  # For reprocess mode
DEFAULT_CONFIGS = []  # For batch mode (list of config names)
DEFAULT_CROSS_CONFIG = None  # For cross-analysis mode
# ============================================


def load_cross_analysis_config(config_name):
    """
    Load a cross-analysis configuration file.
    
    Args:
        config_name (str): Name of the cross-analysis config (without .yaml extension)
        
    Returns:
        dict: Configuration dictionary
    """
    # Check if it's a full path
    if os.path.isfile(config_name):
        config_path = config_name
    else:
        # Look in configs/cross_analysis/
        config_path = os.path.join(os.getcwd(), "configs", "cross_analysis", f"{config_name}.yaml")
        
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Cross-analysis config not found: {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_simulate_mode(config_name):
    """
    Run a simulation and automatically post-process results.
    
    Args:
        config_name (str): Name of the experiment config
    """
    print("="*80)
    print("🚀 MODE: SIMULATE")
    print("="*80)
    
    # Load configuration
    config = load_config(config_name)
    
    # Run simulation
    runner = SimulationRunner(config)
    raw_results_dir = runner.run()
    
    # Auto post-process if configured
    if 'post_processing' in config:
        print("\n" + "="*80)
        print("🔄 Starting automatic post-processing...")
        print("="*80)
        
        processor = SingleSimProcessor(raw_results_dir, config['post_processing'])
        processor.process()
    else:
        print("\n⚠️  No post_processing section found in config. Skipping post-processing.")
        print("💡 Add a 'post_processing' section to your config for automatic post-processing.")
    
    return 0


def run_reprocess_mode(config_name, results_dir):
    """
    Re-process existing simulation results.
    
    Args:
        config_name (str): Name of the experiment config (for post-processing params)
        results_dir (str): Path or name of the results directory to reprocess
    """
    print("="*80)
    print("🔄 MODE: REPROCESS")
    print("="*80)
    
    # Resolve results directory path
    if os.path.isabs(results_dir):
        # Absolute path provided
        full_results_dir = results_dir
    else:
        # Assume it's a directory name in _results/raw_results/
        base_raw_results_dir = os.path.join(os.getcwd(), "_results", "raw_results")
        full_results_dir = os.path.join(base_raw_results_dir, results_dir)
    
    if not os.path.isdir(full_results_dir):
        print(f"❌ ERROR: Results directory not found: {full_results_dir}")
        return 1
    
    print(f"📁 Results Directory: {full_results_dir}")
    
    # Load configuration (only need post_processing section)
    config = load_config(config_name)
    
    if 'post_processing' not in config:
        print("❌ ERROR: Config must contain a 'post_processing' section for reprocessing.")
        return 1
    
    # Process results
    processor = SingleSimProcessor(full_results_dir, config['post_processing'])
    processor.process()
    
    return 0


def run_cross_analysis_mode(config_name):
    """
    Perform cross-simulation analysis.
    
    Args:
        config_name (str): Name of the cross-analysis config
    """
    print("="*80)
    print("📊 MODE: CROSS-ANALYSIS")
    print("="*80)
    
    # Load cross-analysis configuration
    config = load_cross_analysis_config(config_name)
    
    # Extract configuration
    analysis_name = config.get('analysis_name', 'cross_simulation_analysis')
    result_directories = config.get('result_directories', [])
    output_dir = config.get('output_dir', None)
    
    if not result_directories:
        print("❌ ERROR: No result_directories specified in config.")
        return 1
    
    print(f"📊 Analysis Name: {analysis_name}")
    print(f"📁 Number of Results: {len(result_directories)}")
    
    # Resolve result directory paths
    base_raw_results_dir = os.path.join(os.getcwd(), "_results", "raw_results")
    resolved_dirs = []
    
    for result_dir in result_directories:
        if os.path.isabs(result_dir):
            resolved_dirs.append(result_dir)
        else:
            resolved_dirs.append(os.path.join(base_raw_results_dir, result_dir))
    
    # Check that all directories exist
    missing_dirs = [d for d in resolved_dirs if not os.path.isdir(d)]
    if missing_dirs:
        print("\n❌ ERROR: The following result directories were not found:")
        for d in missing_dirs:
            print(f"   • {d}")
        return 1
    
    # Run cross-analysis
    analyzer = CrossSimProcessor(
        result_directories=resolved_dirs,
        base_output_dir=output_dir,
        analysis_name=analysis_name
    )
    analyzer.analyze()
    
    return 0


def run_batch_mode(config_names):
    """
    Run multiple simulations sequentially.
    
    Args:
        config_names (list): List of config names to run
    """
    print("="*80)
    print("📦 MODE: BATCH")
    print("="*80)
    print(f"📋 Number of Configs: {len(config_names)}")
    print("="*80)
    
    results = []
    
    for i, config_name in enumerate(config_names, 1):
        print(f"\n{'='*80}")
        print(f"🔄 BATCH ITEM {i}/{len(config_names)}: {config_name}")
        print("="*80)
        
        try:
            result = run_simulate_mode(config_name)
            results.append((config_name, result))
            
            if result != 0:
                print(f"\n⚠️  Warning: Config '{config_name}' completed with errors")
        except Exception as e:
            print(f"\n❌ ERROR processing config '{config_name}': {e}")
            results.append((config_name, -1))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BATCH SUMMARY")
    print("="*80)
    
    for config_name, result in results:
        status = "✅ Success" if result == 0 else "❌ Failed"
        print(f"  {status}: {config_name}")
    
    print("="*80)
    
    return 0


def main():
    """Main entry point for the simulator."""
    parser = argparse.ArgumentParser(
        description='Chiplet System Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default=DEFAULT_MODE,
        choices=['simulate', 'reprocess', 'cross-analysis', 'batch'],
        help='Operation mode (default: %(default)s)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG,
        help='Config name for simulate/reprocess modes (default: %(default)s)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help='Results directory name or path for reprocess mode'
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        default=','.join(DEFAULT_CONFIGS),
        help='Comma-separated list of config names for batch mode'
    )
    
    parser.add_argument(
        '--cross-config',
        type=str,
        default=DEFAULT_CROSS_CONFIG,
        help='Cross-analysis config name (default uses --config)'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    try:
        if args.mode == 'simulate':
            return run_simulate_mode(args.config)
        
        elif args.mode == 'reprocess':
            if not args.results_dir:
                print("❌ ERROR: --results-dir is required for reprocess mode")
                return 1
            return run_reprocess_mode(args.config, args.results_dir)
        
        elif args.mode == 'cross-analysis':
            cross_config = args.cross_config if args.cross_config else args.config
            return run_cross_analysis_mode(cross_config)
        
        elif args.mode == 'batch':
            if not args.configs:
                print("❌ ERROR: --configs is required for batch mode")
                return 1
            config_names = [c.strip() for c in args.configs.split(',') if c.strip()]
            if not config_names:
                print("❌ ERROR: No valid config names provided")
                return 1
            return run_batch_mode(config_names)
        
        else:
            print(f"❌ ERROR: Unknown mode: {args.mode}")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
