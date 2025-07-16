#!/bin/bash

# Personality Predictor Web Application Startup Script

echo "🧠 Starting Personality Predictor Web Application..."
echo "=================================================="

# Change to backend directory
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if data files exist
if [ ! -f "train.csv" ] || [ ! -f "test.csv" ]; then
    echo "❌ Training data files not found. Please ensure train.csv and test.csv are in the backend directory."
    exit 1
fi

# Check if frontend files exist
if [ ! -d "../frontend/templates" ] || [ ! -d "../frontend/static" ]; then
    echo "❌ Frontend files not found. Please ensure frontend directory exists with templates and static folders."
    exit 1
fi

# Start the application
echo "🚀 Starting Flask application..."
echo "📍 The application will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

python app.py 