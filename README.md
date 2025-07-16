# 🧠 Personality Predictor AI

A machine learning web application that predicts personality traits based on user behavior and preferences. Built with Flask, CatBoost, and modern web technologies.

![Personality Predictor](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Advanced ML Model**: Uses CatBoost algorithm for accurate personality predictions
- **Web Interface**: Clean, responsive UI for easy interaction
- **RESTful API**: JSON endpoints for programmatic access
- **Real-time Predictions**: Instant personality trait analysis
- **Feature Engineering**: Sophisticated data preprocessing and feature selection
- **Model Persistence**: Trained models saved for quick loading

## 🏗️ Architecture

```
playground-series-s5e7/
├── backend/              # Flask API + ML Models
│   ├── app.py           # Main Flask application
│   ├── personality_prediction.py  # ML prediction logic
│   ├── model.joblib     # Trained CatBoost model
│   ├── scaler.joblib    # Feature scaler
│   ├── feature_columns.joblib  # Feature column definitions
│   ├── target_encoder.joblib   # Target encoder
│   └── requirements.txt  # Python dependencies
├── frontend/            # Web UI
│   ├── templates/       # HTML templates
│   │   ├── index.html   # Main prediction interface
│   │   └── about.html   # About page
│   └── static/          # CSS, JS, and assets
│       ├── css/
│       └── js/
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mithila17-UX/Personality_AI.git
   cd Personality_AI
   ```

2. **Setup the backend**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Web Interface: http://localhost:5000
   - API Endpoint: http://localhost:5000/predict

## 📊 Model Performance

The CatBoost model achieves excellent performance on personality prediction:

- **Accuracy**: High prediction accuracy across multiple personality dimensions
- **Feature Importance**: Optimized feature selection for robust predictions
- **Scalability**: Fast inference for real-time predictions

## 🔧 API Usage

### Web Interface
Visit http://localhost:5000 to use the interactive web interface.

### REST API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_spent_alone": 6,
    "stage_fear": 1,
    "social_media_usage": 4,
    "decision_making": 3,
    "risk_taking": 2,
    "leadership_qualities": 4,
    "emotional_stability": 5,
    "creativity_level": 3,
    "analytical_thinking": 4,
    "teamwork_preference": 3
  }'
```

### Response Format
```json
{
  "prediction": "INTJ",
  "confidence": 0.85,
  "personality_traits": {
    "introversion": 0.8,
    "intuition": 0.7,
    "thinking": 0.6,
    "judging": 0.9
  },
  "description": "Architect - Imaginative and strategic thinkers..."
}
```

## 🛠️ Development

### Project Structure
- **Backend**: Flask application with ML models
- **Frontend**: HTML/CSS/JavaScript interface
- **Models**: CatBoost trained models for personality prediction
- **Data**: Training and validation datasets

### Key Files
- `backend/app.py`: Main Flask application
- `backend/personality_prediction.py`: ML prediction logic
- `frontend/templates/index.html`: Main web interface
- `frontend/static/css/style.css`: Styling
- `frontend/static/js/script.js`: Frontend functionality

### Testing
```bash
cd backend
python test_app.py
```

## 📈 Model Training

The model was trained on a comprehensive dataset of personality indicators:

1. **Data Preprocessing**: Feature engineering and scaling
2. **Model Training**: CatBoost algorithm with hyperparameter tuning
3. **Validation**: Cross-validation for robust performance
4. **Feature Selection**: Optimized feature set for prediction accuracy

## 🚀 Deployment

### Local Development
```bash
cd backend
./start_app.sh
```

### Production Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:
- Docker deployment
- Cloud platforms (Heroku, AWS, Google Cloud)
- Nginx configuration
- SSL/TLS setup

## 📊 Features Analyzed

The model analyzes various personality indicators:

- **Social Behavior**: Time spent alone, social media usage
- **Risk Assessment**: Risk-taking tendencies, decision-making style
- **Leadership**: Leadership qualities, teamwork preferences
- **Cognitive Style**: Analytical thinking, creativity levels
- **Emotional Traits**: Emotional stability, stage fear

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CatBoost team for the excellent ML library
- Flask community for the web framework
- Open source contributors and the ML community

## 📞 Contact

- **GitHub**: [@Mithila17-UX](https://github.com/Mithila17-UX)
- **Project Link**: https://github.com/Mithila17-UX/Personality_AI

---

⭐ **Star this repository if you find it helpful!** ⭐ 