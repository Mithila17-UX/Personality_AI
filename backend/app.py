from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the backend directory to the path so we can import personality_predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personality_prediction import PersonalityPredictor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Global variables to store the model and preprocessing objects
model = None
scaler = None
target_encoder = None
feature_columns = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, target_encoder, feature_columns
    
    # Check if model files exist, if not train the model
    if not os.path.exists('model.joblib'):
        print("Training model...")
        predictor = PersonalityPredictor()
        predictor.load_data()
        predictor.preprocess_data()
        predictor.prepare_features()
        predictor.model_scores = predictor.train_models()
        
        # Save the best model and preprocessing objects
        best_model_name = max(predictor.model_scores, key=predictor.model_scores.get)
        model = predictor.models[best_model_name]
        scaler = predictor.scaler
        target_encoder = predictor.target_encoder
        feature_columns = predictor.feature_columns
        
        # Save model and preprocessing objects
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(target_encoder, 'target_encoder.joblib')
        joblib.dump(feature_columns, 'feature_columns.joblib')
    else:
        # Load saved model and preprocessing objects
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        target_encoder = joblib.load('target_encoder.joblib')
        feature_columns = joblib.load('feature_columns.joblib')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Time_spent_Alone': float(data['time_spent_alone']),
            'Stage_fear': int(data['stage_fear']),
            'Social_event_attendance': float(data['social_event_attendance']),
            'Going_outside': float(data['going_outside']),
            'Drained_after_socializing': int(data['drained_after_socializing']),
            'Friends_circle_size': float(data['friends_circle_size']),
            'Post_frequency': float(data['post_frequency'])
        }])
        
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
        
        # Select features for prediction
        X = input_data[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_scaled)
        prediction = target_encoder.inverse_transform([prediction_proba.argmax()])[0]
        confidence = prediction_proba.max()
        
        # Get personality insights
        insights = get_personality_insights(input_data.iloc[0], prediction)
        
        # Fix probability mapping based on target encoder classes
        # Target encoder classes: ['Extrovert', 'Introvert']
        # So prediction_proba[0][0] = Extrovert probability, prediction_proba[0][1] = Introvert probability
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'extrovert_probability': float(prediction_proba[0][0]),
            'introvert_probability': float(prediction_proba[0][1]),
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_personality_insights(data, prediction):
    """Generate insights based on the prediction and input data"""
    insights = []
    
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
    
    return insights

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load model on startup
    load_model()
    print("Model loaded successfully!")
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000) 