# ML Models Directory 🤖

This directory contains the machine learning model implementations for the ML Model Runner application. Each model runs as an independent Flask service with its own virtual environment and dependencies.

## 📁 Directory Structure

```
ml_models/
├── ffnn/           # Feed-Forward Neural Network
├── xgboost/        # XGBoost Model
├── catboost/       # CatBoost Model
├── lightgbm/       # LightGBM Model (Available)
├── fcnn/           # Fully Connected Neural Network (FCNN)
└── README.md       # This file
```

## 🏗️ Architecture

Each model follows a consistent architecture pattern:

### Common Components
- **Service File**: `*_service.py` - Flask API server
- **Setup Script**: `setup.py` - Automated dependency installation
- **Virtual Environment**: `venv/` - Isolated Python environment
- **Model Files**: Pre-trained model weights and data

### Communication Protocol
- **Port Assignment**: Each model has a dedicated port
- **REST API**: Standard HTTP endpoints for predictions
- **JSON Format**: Request/response data in JSON format

## 🚀 Quick Setup

### Setup All Models
```bash
# From the project root
cd lib/ml_models

# Setup FFNN (Neural Network)
cd ffnn && python setup.py && cd ..

# Setup XGBoost
cd xgboost && python setup.py && cd ..

# Setup CatBoost  
cd catboost && python setup.py && cd ..

# Setup FCNN (Fully Connected Neural Network)
cd fcnn && python setup.py && cd ..
```

### Individual Model Setup
```bash
# Navigate to specific model directory
cd lib/ml_models/[model_name]

# Run setup script
python setup.py

# Start the service (optional - app does this automatically)
python [model_name]_service.py
```

## 🔧 Model Details

### 1. FFNN (Feed-Forward Neural Network)
- **Directory**: `ffnn/`
- **Port**: 5002
- **Technology**: TensorFlow 2.20+ (tf-nightly)
- **Python Version**: 3.7+ (3.13 supported)
- **Dependencies**: `tf-nightly`, `protobuf<5`, `pandas`, `numpy`, `scikit-learn`
- **Model File**: `nn_model.h5`
- **Special Notes**: Uses tf-nightly for Python 3.13 compatibility

### 2. XGBoost
- **Directory**: `xgboost/`
- **Port**: 5000
- **Technology**: XGBoost + scikit-learn
- **Python Version**: 3.7+
- **Dependencies**: `xgboost`, `pandas`, `numpy`, `scikit-learn`
- **Model File**: Pre-trained XGBoost model

### 3. CatBoost
- **Directory**: `catboost/`
- **Port**: 5001
- **Technology**: CatBoost
- **Python Version**: 3.7+
- **Dependencies**: `catboost`, `pandas`, `numpy`, `scikit-learn`
- **Model File**: Pre-trained CatBoost model

### 4. LightGBM
- **Directory**: `lightgbm/`
- **Port**: 5003
- **Technology**: LightGBM
- **Status**: Available (Setup Required)
- **Dependencies**: `lightgbm`, `pandas`, `numpy`, `scikit-learn`

### 5. FCNN (Fully Connected Neural Network)
- **Directory**: `fcnn/`
- **Port**: 5004
- **Technology**: TensorFlow/Keras with Lasso feature selection
- **Python Version**: 3.7+ (3.13 supported)
- **Dependencies**: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `joblib`
- **Model Files**: `mlp_delta_e_model_lasso.h5`, `mlp_scaler_lasso.joblib`, `mlp_model_features_lasso.joblib`
- **Special Notes**: Deep learning model with automatic feature selection

## 🔄 API Endpoints

Each model service provides the following standard endpoints:

### Health Check
```http
GET /health
Response: {"status": "healthy", "model": "model_name"}
```

### Predict
```http
POST /predict
Content-Type: application/json

Request Body:
{
  "features": [feature1, feature2, ...],
  "iterations": 1000
}

Response:
{
  "prediction": result_value,
  "model": "model_name",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🐛 Troubleshooting

### Common Issues

#### Virtual Environment Creation Failed
```bash
# Ensure Python has venv module
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

#### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :5001  # Windows
lsof -i :5001                 # macOS/Linux

# Kill the process if needed
taskkill /F /PID <PID>        # Windows
kill -9 <PID>                 # macOS/Linux
```

#### TensorFlow Installation Issues (FFNN)
```bash
# For Python 3.13, use tf-nightly
pip install tf-nightly protobuf<5

# For older Python versions
pip install tensorflow protobuf<5
```

#### Missing Dependencies
```bash
# Re-run setup script
cd lib/ml_models/[model_name]
python setup.py

# Or install manually
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 🔧 Development

### Adding a New Model

1. **Create Directory**: `lib/ml_models/new_model/`
2. **Service File**: Implement Flask API in `new_model_service.py`
3. **Setup Script**: Create `setup.py` for dependencies
4. **Model Registration**: Add to `lib/models/model_factory.dart`
5. **Dart Implementation**: Create `lib/models/new_model.dart`

### Model Service Template
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "new_model"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Implement prediction logic
    result = your_prediction_function(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='localhost', port=5004, debug=False)
```

## 📊 Performance Notes

- **Startup Time**: Models may take 10-30 seconds to fully initialize
- **Memory Usage**: Each model maintains its own memory space
- **Concurrent Requests**: Models can handle multiple simultaneous predictions
- **Resource Management**: Virtual environments prevent dependency conflicts

## 🔍 Monitoring

The Flutter app provides real-time monitoring for all models:
- Health status indicators
- Automatic restart capabilities  
- Error logging and reporting
- Performance metrics

## 📝 Maintenance

### Regular Tasks
- Update dependencies periodically
- Monitor disk space (models and datasets can be large)
- Check logs for errors or performance issues
- Backup trained model files

### Updates
When updating models:
1. Stop the service
2. Update the model files
3. Restart the service
4. Verify health check passes

---

**For more information, see the main project README or individual model documentation.** 