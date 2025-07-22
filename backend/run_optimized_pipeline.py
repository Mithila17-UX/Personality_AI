#!/usr/bin/env python3
"""
Simple script to run the hyperoptimized personality prediction pipeline
with CatBoost optimization skipped to avoid timeout issues.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personality_prediction_hyperoptimized_simplified import HyperOptimizedPersonalityPredictor

def main():
    """Run the hyperoptimized pipeline with CatBoost optimization skipped"""
    print("Starting hyperoptimized personality prediction pipeline...")
    print("Note: CatBoost optimization is skipped to avoid timeout issues.")
    print("=" * 60)
    
    try:
        # Create and run the hyperoptimized predictor
        predictor = HyperOptimizedPersonalityPredictor()
        submission = predictor.run_hyperoptimized_pipeline()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Submission file created: {submission.shape[0]} predictions")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 