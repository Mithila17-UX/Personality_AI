# Deployment Guide - Personality Predictor

This guide explains how to deploy the Personality Predictor application with its separated backend and frontend structure.

## 🏗️ Architecture Overview

```
playground-series-s5e7/
├── backend/          # Flask API + ML Models
└── frontend/         # HTML/CSS/JS UI
```

## 🚀 Quick Deployment

### 1. Local Development
```bash
# Navigate to backend
cd backend

# Start the application
./start_app.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 2. Access the Application
- **URL**: http://localhost:5000
- **API**: http://localhost:5000/predict
- **About**: http://localhost:5000/about

## 🔧 Development Workflow

### Backend Development
```bash
cd backend
# Make changes to app.py or personality_prediction.py
python app.py
```

### Frontend Development
```bash
cd frontend
# Make changes to templates/ or static/
# The Flask app automatically serves updated files
```

### Testing
```bash
cd backend
python test_app.py
```

## 📦 Production Deployment

### Option 1: Single Server Deployment

#### Prerequisites
- Python 3.7+
- Nginx (optional, for reverse proxy)
- Gunicorn (for production WSGI server)

#### Setup
```bash
# 1. Clone/Download the project
git clone <repository-url>
cd playground-series-s5e7

# 2. Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# 3. Start with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Nginx Configuration (Optional)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/playground-series-s5e7/frontend/static/;
    }
}
```

### Option 2: Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set working directory
WORKDIR /app/backend

# Expose port
EXPOSE 5000

# Start the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  personality-predictor:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./backend:/app/backend
      - ./frontend:/app/frontend
```

### Option 3: Cloud Deployment

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT backend.app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS EC2
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip nginx

# Setup application
cd /var/www/
git clone <repository-url>
cd playground-series-s5e7/backend
pip3 install -r requirements.txt
pip3 install gunicorn

# Create systemd service
sudo nano /etc/systemd/system/personality-predictor.service
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/personality-predictor
gcloud run deploy personality-predictor --image gcr.io/PROJECT_ID/personality-predictor --platform managed
```

## 🔒 Security Considerations

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export DATABASE_URL=your-database-url
```

### SSL/TLS Configuration
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
    }
}
```

### Rate Limiting
```python
# In app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

## 📊 Monitoring

### Health Checks
```bash
# Test application health
curl http://localhost:5000/
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"time_spent_alone": 6, "stage_fear": 1, ...}'
```

### Logging
```python
# In app.py
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health')
def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}
```

### Performance Monitoring
```bash
# Monitor application performance
htop
ps aux | grep python
netstat -tulpn | grep 5000
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill process
   kill -9 <PID>
   ```

2. **Permission Denied**
   ```bash
   # Fix file permissions
   chmod +x backend/start_app.sh
   chmod 755 frontend/static/
   ```

3. **Model Loading Issues**
   ```bash
   # Retrain models
   cd backend
   rm *.joblib
   python app.py
   ```

4. **Static Files Not Loading**
   ```bash
   # Check frontend structure
   ls -la frontend/static/
   ls -la frontend/templates/
   ```

### Debug Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Multiple instances
gunicorn -w 8 -b 0.0.0.0:5000 app:app
```

### Load Balancing
```nginx
upstream personality_predictor {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    location / {
        proxy_pass http://personality_predictor;
    }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy Personality Predictor

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          # Deployment commands
```

---

**Your Personality Predictor is now ready for production deployment! 🚀✨** 