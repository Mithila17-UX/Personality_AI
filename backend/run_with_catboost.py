#!/usr/bin/env python3
"""
Script to run the hyperoptimized personality prediction pipeline
with CatBoost optimization included (may take longer).
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personality_prediction_hyperoptimized_simplified import HyperOptimizedPersonalityPredictor

def main():
    """Run the hyperoptimized pipeline with CatBoost optimization"""
    print("Starting hyperoptimized personality prediction pipeline with CatBoost...")
    print("Warning: This may take longer due to CatBoost optimization.")
    print("=" * 60)
    
    try:
        # Create and run the hyperoptimized predictor with CatBoost
        predictor = HyperOptimizedPersonalityPredictor()
        submission = predictor.run_hyperoptimized_pipeline_with_catboost()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Submission file created: {submission.shape[0]} predictions")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        print("If CatBoost optimization was taking too long, try running without it:")
        print("python run_optimized_pipeline.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        print("If CatBoost optimization was causing issues, try running without it:")
        print("python run_optimized_pipeline.py")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 