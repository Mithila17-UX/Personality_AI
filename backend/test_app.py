#!/usr/bin/env python3
"""
Test script for Personality Predictor Web Application
"""

import requests
import json
import time
import sys
import os

def test_app():
    """Test the Flask application"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Personality Predictor Web Application")
    print("=" * 50)
    
    # Test 1: Check if the app is running
    print("1. Testing if application is running...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Application is running successfully!")
        else:
            print(f"❌ Application returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Application is not running. Please start the app first:")
        print("   cd backend && python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to application: {e}")
        return False
    
    # Test 2: Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    
    # Sample data for testing
    test_data = {
        "time_spent_alone": 6,
        "stage_fear": 1,
        "social_event_attendance": 3,
        "going_outside": 2,
        "drained_after_socializing": 1,
        "friends_circle_size": 15,
        "post_frequency": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Prediction endpoint working correctly!")
                print(f"   Prediction: {result.get('prediction')}")
                print(f"   Confidence: {result.get('confidence', 0):.2%}")
                print(f"   Introvert Probability: {result.get('introvert_probability', 0):.2%}")
                print(f"   Extrovert Probability: {result.get('extrovert_probability', 0):.2%}")
                print(f"   Insights: {len(result.get('insights', []))} insights provided")
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Prediction endpoint returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False
    
    # Test 3: Test about page
    print("\n3. Testing about page...")
    try:
        response = requests.get(f"{base_url}/about", timeout=5)
        if response.status_code == 200:
            print("✅ About page is accessible!")
        else:
            print(f"❌ About page returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing about page: {e}")
        return False
    
    print("\n🎉 All tests passed! The application is working correctly.")
    print("\n📱 You can now:")
    print("   - Open http://localhost:5000 in your browser")
    print("   - Fill out the personality questionnaire")
    print("   - Get instant AI-powered personality insights")
    
    return True

if __name__ == "__main__":
    success = test_app()
    sys.exit(0 if success else 1) 