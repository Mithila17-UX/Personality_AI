# Replit Deployment Guide

## Overview
This guide explains how to deploy the Personality AI web application to Replit.

## Prerequisites
- A Replit account (free at replit.com)
- Your project files ready for deployment

## Deployment Steps

### 1. Create a New Replit Project
1. Go to [replit.com](https://replit.com) and sign in
2. Click "Create Repl"
3. Choose "Python" as the language
4. Give your project a name (e.g., "personality-ai")
5. Click "Create Repl"

### 2. Upload Your Project Files
You have two options:

#### Option A: Import from GitHub
1. In your Replit workspace, click the "Version Control" icon (Git icon)
2. Click "Import from GitHub"
3. Enter your repository URL: `https://github.com/Mithila17-UX/Personality_AI.git`
4. Select the branch: `Hypertuning-V3`
5. Click "Import from GitHub"

#### Option B: Manual Upload
1. Upload all project files to your Replit workspace:
   - `app.py`
   - `requirements.txt`
   - `model.joblib`
   - `scaler.joblib`
   - `target_encoder.joblib`
   - `feature_columns.joblib`
   - `static/` folder
   - `templates/` folder
   - `.replit` (created automatically)
   - `pyproject.toml`

### 3. Install Dependencies
1. In the Replit shell, run:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Run the Application
1. Click the "Run" button in Replit
2. The application will start and show a webview
3. Your app will be available at the provided URL

## Configuration Files

### .replit
This file tells Replit how to run your application:
```
language = "python3"
run = "python app.py"
```

### pyproject.toml
This file provides better dependency management and project metadata.

## Troubleshooting

### Common Issues:

1. **Model files not found**
   - Ensure all `.joblib` files are uploaded
   - Check file paths in `app.py`

2. **Port issues**
   - Replit automatically assigns a port
   - The app uses `os.environ.get("PORT", 3000)` to handle this

3. **Dependencies not installed**
   - Run `pip install -r requirements.txt` in the shell
   - Check the "Packages" tab in Replit

4. **Application not starting**
   - Check the console for error messages
   - Ensure `app.py` is in the root directory

## Features
- ✅ Flask web application
- ✅ Machine learning model integration
- ✅ Static file serving (CSS, JS)
- ✅ Template rendering
- ✅ RESTful API endpoints
- ✅ Health check endpoint

## API Endpoints
- `GET /` - Main application page
- `POST /predict` - Personality prediction endpoint
- `GET /about` - About page
- `GET /health` - Health check endpoint

## Environment Variables
The application automatically detects Replit's environment:
- `PORT` - Automatically set by Replit
- `HOST` - Set to "0.0.0.0" for external access

## Security Notes
- The application runs in Replit's secure environment
- No sensitive data is exposed
- Model files are included in the repository

## Support
If you encounter issues:
1. Check the Replit console for error messages
2. Verify all files are uploaded correctly
3. Ensure dependencies are installed
4. Check the health endpoint: `/health` 