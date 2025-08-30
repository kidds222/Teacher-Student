#!/usr/bin/env python3
"""
Advanced Dynamic Teaching System - Simplified entry point
Run the dynamic lifelong learning experiment. All configurations are controlled via files in the config/ directory.
"""

import os
import sys
import argparse

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the simplified running system
try:
    from experiments.run import main as run_experiment
    EXPERIMENTS_AVAILABLE = True
except ImportError:
    print(" experiments.run module is not available")
    EXPERIMENTS_AVAILABLE = False


def main():
    """Main function - simplified version that directly runs the experiment"""
    parser = argparse.ArgumentParser(description="Advanced Dynamic Teaching System")
    parser.add_argument('--enable_mixed_samples', action='store_true', 
                       help='Enable mixed sample generation (will be triggered when running Task 2)')
    parser.add_argument('--mixed_samples_only', action='store_true',
                       help='Only generate mixed samples (a checkpoint path is required)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path (used for mixed sample generation)')
    parser.add_argument('--output_dir', type=str, default='mixed_samples',
                       help='Output directory for mixed samples')
    
    args = parser.parse_args()
    
    print(" AdvancedDynamicTeaching - Dynamic Lifelong Learning System")
    print("=" * 60)
    print(" Configuration: edit files under config/")
    print("   - config/teacher_config.py   (Teacher network configuration)")
    print("   - config/student_config.py   (Student network configuration)")  
    print("   - config/experiment_config.py (Experiment flow configuration)")
    print("=" * 60)
    
    # Check mixed-samples-only mode
    if args.mixed_samples_only:
        if not args.checkpoint:
            print(" A checkpoint path is required for --mixed_samples_only mode")
            print(" Usage: python main.py --mixed_samples_only --checkpoint path/to/checkpoint.pth")
            return
        
        print("Mixed samples only mode")
        print(f" Checkpoint: {args.checkpoint}")
        print(f" Output directory: {args.output_dir}")
        
        try:
            print("WARNING: Mixed sample generator is not available in this version")
            print(f" Skipping mixed sample generation")
        except Exception as e:
            print(f" Mixed sample generation failed: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Informational message for mixed sample feature
    if args.enable_mixed_samples:
        print(" Mixed sample generation is enabled")
        print("   - Mixed samples will be generated during Task 2")
        print("   - Includes Teacher mix, Student mix, and interpolation sequence")
        print("   - Mixed samples will be saved under samples/mixed_samples/")
    
    print(" Starting experiment...\n")
    
    if not EXPERIMENTS_AVAILABLE:
        print(" Experiment module is not available, please check experiments/run.py")
        return
    
    # Directly run the experiment
    try:
        results = run_experiment()
        print("\n Experiment completed!")
        
        # If mixed sample feature is enabled, show related info
        if args.enable_mixed_samples:
            print("\n Mixed sample feature is enabled")
            print(" Mixed samples saved at: results/*/samples/mixed_samples/")
            print(" Review generated mixed samples to understand knowledge fusion across tasks")
        
        return results
    except KeyboardInterrupt:
        print("\n Experiment interrupted by user")
        return None
    except Exception as e:
        print(f" Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 