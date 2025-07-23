from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
from pathlib import Path

# Replit-specific configurations
os.environ['PYTHONPATH'] = os.getcwd()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading will be handled by joblib directly

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store the model and preprocessing objects
model = None
scaler = None
target_encoder = None
feature_columns = None
model_loaded = False

def load_model():
    """Load the trained model and preprocessing objects with error handling"""
    global model, scaler, target_encoder, feature_columns, model_loaded
    
    try:
        # Get the current directory and check for model files
        current_dir = Path(__file__).parent
        logger.info(f"Current directory: {current_dir}")
        
        # Check if model files exist in current directory
        model_path = current_dir / 'model.joblib'
        scaler_path = current_dir / 'scaler.joblib'
        target_encoder_path = current_dir / 'target_encoder.joblib'
        feature_columns_path = current_dir / 'feature_columns.joblib'
        
        logger.info(f"Checking for model files in: {current_dir}")
        logger.info(f"Model path exists: {model_path.exists()}")
        logger.info(f"Scaler path exists: {scaler_path.exists()}")
        logger.info(f"Target encoder path exists: {target_encoder_path.exists()}")
        logger.info(f"Feature columns path exists: {feature_columns_path.exists()}")
        
        # For Replit, also check if files are in the root directory
        if not model_path.exists():
            # Try alternative paths for Replit
            alt_paths = [
                Path.cwd() / 'model.joblib',
                Path.cwd() / 'scaler.joblib',
                Path.cwd() / 'target_encoder.joblib',
                Path.cwd() / 'feature_columns.joblib'
            ]
            logger.info(f"Trying alternative paths for Replit: {alt_paths}")
            
            if not any(p.exists() for p in alt_paths):
                logger.error("Model files not found. Cannot proceed without trained model.")
                model_loaded = False
                return
            else:
                # Use alternative paths
                model_path = Path.cwd() / 'model.joblib'
                scaler_path = Path.cwd() / 'scaler.joblib'
                target_encoder_path = Path.cwd() / 'target_encoder.joblib'
                feature_columns_path = Path.cwd() / 'feature_columns.joblib'
        
        # Load saved model and preprocessing objects
        logger.info("Loading pre-trained model...")
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            target_encoder = joblib.load(target_encoder_path)
            feature_columns = joblib.load(feature_columns_path)
            logger.info("Model loaded successfully!")
        except Exception as load_error:
            logger.error(f"Failed to load model files: {load_error}")
            model_loaded = False
            return
        
        model_loaded = True
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False

def validate_input_data(data):
    """Validate and sanitize input data"""
    required_fields = [
        'time_spent_alone',
        'stage_fear', 
        'social_event_attendance',
        'going_outside',
        'drained_after_socializing',
        'friends_circle_size',
        'post_frequency'
    ]
    
    # Check if data is None or empty
    if not data:
        return False, "No data received"
    
    # Check for required fields
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate data types and ranges
    try:
        # Convert and validate numeric fields
        time_spent_alone = float(data['time_spent_alone'])
        if not (0 <= time_spent_alone <= 24):
            return False, "time_spent_alone must be between 0 and 24"
        
        stage_fear = int(data['stage_fear'])
        if stage_fear not in [0, 1]:
            return False, "stage_fear must be 0 or 1"
        
        social_event_attendance = float(data['social_event_attendance'])
        if not (0 <= social_event_attendance <= 20):
            return False, "social_event_attendance must be between 0 and 20"
        
        going_outside = float(data['going_outside'])
        if not (0 <= going_outside <= 14):
            return False, "going_outside must be between 0 and 14"
        
        drained_after_socializing = int(data['drained_after_socializing'])
        if drained_after_socializing not in [0, 1]:
            return False, "drained_after_socializing must be 0 or 1"
        
        friends_circle_size = float(data['friends_circle_size'])
        if not (0 <= friends_circle_size <= 100):
            return False, "friends_circle_size must be between 0 and 100"
        
        post_frequency = float(data['post_frequency'])
        if not (0 <= post_frequency <= 20):
            return False, "post_frequency must be between 0 and 20"
        
        return True, {
            'Time_spent_Alone': time_spent_alone,
            'Stage_fear': stage_fear,
            'Social_event_attendance': social_event_attendance,
            'Going_outside': going_outside,
            'Drained_after_socializing': drained_after_socializing,
            'Friends_circle_size': friends_circle_size,
            'Post_frequency': post_frequency
        }
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with comprehensive error handling"""
    try:
        # Log the request
        logger.info("Received prediction request")
        
        # Check if model is loaded
        if not model_loaded:
            logger.error("Model not loaded - attempting fallback prediction")
            # Try to provide a fallback prediction based on simple rules
            try:
                data = request.get_json()
                logger.info(f"Fallback prediction with data: {data}")
                
                # Simple fallback logic
                time_alone = float(data.get('time_spent_alone', 8))
                stage_fear = int(data.get('stage_fear', 1))
                social_events = float(data.get('social_event_attendance', 3))
                drained = int(data.get('drained_after_socializing', 1))
                
                # Simple introvert/extrovert scoring
                introvert_score = time_alone + (stage_fear * 5) + (drained * 3) - social_events
                
                if introvert_score > 10:
                    prediction = "Introvert"
                    confidence = 0.7
                else:
                    prediction = "Extrovert"
                    confidence = 0.7
                
                return jsonify({
                    'success': True,
                    'result': prediction,
                    'confidence': confidence,
                    'extrovert_probability': 1.0 - confidence if prediction == "Introvert" else confidence,
                    'introvert_probability': confidence if prediction == "Introvert" else 1.0 - confidence,
                    'insights': [f"Fallback prediction based on basic analysis. Model loading failed."],
                    'warning': 'Using fallback prediction - model not available'
                })
            except Exception as fallback_error:
                logger.error(f"Fallback prediction also failed: {fallback_error}")
                return jsonify({
                    'success': False,
                    'error': 'Server error: Model not available. Please try again later.'
                }), 500
        
        # Get and validate input data
        try:
            data = request.get_json()
            logger.info(f"Received data: {data}")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return jsonify({
                'success': False,
                'error': 'Invalid JSON format'
            }), 400
        
        # Validate input data
        is_valid, validation_result = validate_input_data(data)
        if not is_valid:
            logger.error(f"Validation failed: {validation_result}")
            return jsonify({
                'success': False,
                'error': f'Invalid input: {validation_result}'
            }), 400
        
        validated_data = validation_result
        
        # Create input dataframe
        input_data = pd.DataFrame([validated_data])
        
        # Preprocess the input data (similar to training preprocessing)
        # Handle categorical variables
        input_data['Stage_fear'] = input_data['Stage_fear'].map({0: 'No', 1: 'Yes'})
        input_data['Drained_after_socializing'] = input_data['Drained_after_socializing'].map({0: 'No', 1: 'Yes'})
        
        # Create engineered features
        input_data['social_activity_score'] = (
            input_data['Social_event_attendance'] + 
            input_data['Going_outside'] + 
            input_data['Post_frequency']
        )
        
        input_data['introversion_score'] = (
            input_data['Time_spent_Alone'] + 
            input_data['Stage_fear'].map({'No': 0, 'Yes': 1}) * 5 + 
            input_data['Drained_after_socializing'].map({'No': 0, 'Yes': 1}) * 3
        )
        
        input_data['social_efficiency'] = input_data['social_activity_score'] / (input_data['Friends_circle_size'] + 1)
        
        input_data['alone_ratio'] = input_data['Time_spent_Alone'] / (input_data['Time_spent_Alone'] + input_data['social_activity_score'] + 1)
        
        input_data['fear_social_interaction'] = input_data['Stage_fear'].map({'No': 0, 'Yes': 1}) * input_data['Social_event_attendance']
        input_data['drain_social_interaction'] = input_data['Drained_after_socializing'].map({'No': 0, 'Yes': 1}) * input_data['Social_event_attendance']
        
        # Encode categorical variables
        input_data['Stage_fear'] = input_data['Stage_fear'].map({'No': 0, 'Yes': 1})
        input_data['Drained_after_socializing'] = input_data['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
        
        # Ensure we have the correct feature columns
        if feature_columns is None:
            logger.error("Feature columns not available")
            return jsonify({
                'success': False,
                'error': 'Server error: Feature configuration not available'
            }), 500
        
        # Select features for prediction
        try:
            X = input_data[feature_columns]
        except KeyError as e:
            logger.error(f"Missing feature columns: {e}")
            return jsonify({
                'success': False,
                'error': 'Server error: Feature mismatch'
            }), 500
        
        # Scale features
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Server error: Data preprocessing failed'
            }), 500
        
        # Make prediction
        try:
            prediction_proba = model.predict_proba(X_scaled)
            prediction = target_encoder.inverse_transform([prediction_proba.argmax()])[0]
            confidence = prediction_proba.max()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Server error: Prediction failed'
            }), 500
        
        # Get personality insights
        insights = get_personality_insights(input_data.iloc[0], prediction)
        
        # Prepare response
        response = {
            'success': True,
            'result': prediction,  # Changed from 'prediction' to 'result'
            'confidence': float(confidence),
            'extrovert_probability': float(prediction_proba[0][0]),
            'introvert_probability': float(prediction_proba[0][1]),
            'insights': insights
        }
        
        logger.info(f"Prediction successful: {prediction} with confidence {confidence}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {e}")
        return jsonify({
            'success': False,
            'error': 'Server error occurred'
        }), 500

def get_personality_insights(data, prediction):
    """Generate insights based on the prediction and input data"""
    insights = []
    
    try:
        if prediction == 'Introvert':
            if data['Time_spent_Alone'] > 6:
                insights.append("You spend significant time alone, which is typical for introverts.")
            if data['Stage_fear'] == 1:
                insights.append("You experience stage fear, which is common among introverts.")
            if data['Drained_after_socializing'] == 1:
                insights.append("You feel drained after socializing, indicating introverted tendencies.")
            if data['social_activity_score'] < 8:
                insights.append("Your social activity level is relatively low, typical of introverts.")
        else:  # Extrovert
            if data['social_activity_score'] > 12:
                insights.append("You have high social activity, which is characteristic of extroverts.")
            if data['Friends_circle_size'] > 50:
                insights.append("You have a large circle of friends, typical of extroverts.")
            if data['Post_frequency'] > 3:
                insights.append("You frequently share on social media, indicating extroverted behavior.")
            if data['Going_outside'] > 4:
                insights.append("You spend significant time outside, which extroverts typically enjoy.")
        
        # Add general insights
        if data['social_efficiency'] > 0.5:
            insights.append("You maintain good social connections relative to your activity level.")
        else:
            insights.append("You might prefer deeper, fewer relationships over many acquaintances.")
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights = ["Analysis completed successfully."]
    
    return insights

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_files_exist': {
            'model.joblib': Path(__file__).parent / 'model.joblib',
            'scaler.joblib': Path(__file__).parent / 'scaler.joblib',
            'target_encoder.joblib': Path(__file__).parent / 'target_encoder.joblib',
            'feature_columns.joblib': Path(__file__).parent / 'feature_columns.joblib'
        }
    })

if __name__ == "__main__":
    # Load model on startup
    try:
        load_model()
        if model_loaded:
            print("✅ Model loaded successfully!")
        else:
            print("⚠️  Warning: Model failed to load. App will start but predictions may fail.")
        print("🚀 Starting Flask app...")
        
        # Replit-specific startup message
        port = int(os.environ.get("PORT", 3000))
        print(f"🌐 App will be available at: http://localhost:{port}")
        print("📊 Health check available at: /health")
        print("🎯 Main app available at: /")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  Starting app without model...")
    
    # Get port from environment variable (for Replit) or use default
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False) 