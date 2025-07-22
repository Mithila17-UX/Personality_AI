#!/usr/bin/env python3
"""
Test script for the Flask personality prediction app
"""

import json
import requests
import time
import sys

def test_app():
    """Test the Flask app endpoints"""
    
    base_url = "http://localhost:10000"
    
    # Test data
    test_data = {
        "time_spent_alone": 8,
        "stage_fear": 1,
        "social_event_attendance": 3,
        "going_outside": 2,
        "drained_after_socializing": 1,
        "friends_circle_size": 15,
        "post_frequency": 1
    }
    
    print("🧪 Testing Personality Predictor App")
    print("=" * 50)
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
        # Test prediction endpoint
        print("\n2. Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Prediction successful!")
                print(f"   Result: {result.get('result')}")
                print(f"   Confidence: {result.get('confidence'):.2f}")
                print(f"   Extrovert Probability: {result.get('extrovert_probability'):.2f}")
                print(f"   Introvert Probability: {result.get('introvert_probability'):.2f}")
                print(f"   Insights: {len(result.get('insights', []))} insights generated")
            else:
                print(f"❌ Prediction failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        # Test invalid data
        print("\n3. Testing error handling...")
        invalid_data = {"invalid": "data"}
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 400:
            result = response.json()
            print(f"✅ Error handling works: {result.get('error')}")
        else:
            print(f"❌ Error handling failed: {response.status_code}")
            
        print("\n🎉 All tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the app. Make sure it's running on http://localhost:10000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        import joblib
        from pathlib import Path
        
        model_files = ['model.joblib', 'scaler.joblib', 'target_encoder.joblib', 'feature_columns.joblib']
        
        for file in model_files:
            if Path(file).exists():
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                return False
                
        # Try to load the model
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        target_encoder = joblib.load('target_encoder.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        
        print("✅ All model files loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Personality Predictor App Tests")
    
    # Test model loading first
    if not test_model_loading():
        print("\n❌ Model loading tests failed. Cannot proceed with app tests.")
        sys.exit(1)
    
    # Test the app
    if test_app():
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 