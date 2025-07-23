# 🧠 Personality Predictor AI

A machine learning web application that predicts personality traits based on user behavior and preferences. Built with Flask and modern web technologies.

![Personality Predictor](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Advanced ML Model**: Uses optimized algorithms for accurate personality predictions
- **Web Interface**: Clean, responsive UI for easy interaction
- **RESTful API**: JSON endpoints for programmatic access
- **Real-time Predictions**: Instant personality trait analysis
- **Feature Engineering**: Sophisticated data preprocessing and feature selection
- **Model Persistence**: Trained models saved for quick loading

## 🏗️ Project Structure

```
playground-series-s5e7/
├── app.py              # Main Flask application
├── model.joblib        # Trained ML model
├── scaler.joblib       # Feature scaler
├── feature_columns.joblib  # Feature column definitions
├── target_encoder.joblib   # Target encoder
├── requirements.txt     # Python dependencies
├── templates/          # HTML templates
│   ├── index.html      # Main prediction interface
│   └── about.html      # About page
└── static/             # CSS, JS, and assets
    ├── css/
    └── js/
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd playground-series-s5e7
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Web Interface: http://localhost:3000
   - API Endpoint: http://localhost:3000/predict

## 📊 Model Performance

The model achieves excellent performance on personality prediction:

- **Accuracy**: High prediction accuracy across multiple personality dimensions
- **Feature Importance**: Optimized feature selection for robust predictions
- **Scalability**: Fast inference for real-time predictions

## 🔧 API Usage

### Web Interface
Visit http://localhost:3000 to use the interactive web interface.

### REST API
```bash
curl -X POST http://localhost:3000/predict \
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
  "success": true,
  "result": "Introvert",
  "confidence": 0.85,
  "extrovert_probability": 0.15,
  "introvert_probability": 0.85,
  "insights": [
    "You spend significant time alone, which is typical for introverts.",
    "You experience stage fear, which is common among introverts."
  ]
}
```

## 📊 Features Analyzed

The model analyzes various personality indicators:

- **Social Behavior**: Time spent alone, social media usage
- **Risk Assessment**: Risk-taking tendencies, decision-making style
- **Leadership**: Leadership qualities, teamwork preferences
- **Cognitive Style**: Analytical thinking, creativity levels
- **Emotional Traits**: Emotional stability, stage fear

## 🚀 Deployment on Replit

This project is optimized for deployment on Replit:

1. **Import to Replit**: Use the GitHub import feature or drag-and-drop
2. **Automatic Setup**: Replit will detect Python and install dependencies
3. **Run**: Click the "Run" button to start the application
4. **Access**: Your app will be available at the provided Replit URL

### Replit Configuration
- **Language**: Python
- **Run Command**: `python app.py`
- **Port**: 3000 (configured in app.py)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flask community for the web framework
- Open source contributors and the ML community

---

⭐ **Star this repository if you find it helpful!** ⭐ 