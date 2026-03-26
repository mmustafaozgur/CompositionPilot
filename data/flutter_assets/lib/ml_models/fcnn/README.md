# FCNN (Fully Connected Neural Network) Model 🧠

This directory contains the implementation of the Fully Connected Neural Network (FCNN) model for the ML Model Runner application. The FCNN model is a TensorFlow-based deep learning model that predicts delta E values for material compositions.

## 📁 Directory Structure

```
fcnn/
├── fcnn_service.py                      # Flask API server
├── setup.py                            # Automated dependency installation
├── README.md                           # This file
├── venv/                               # Virtual environment (auto-created)
├── __pycache__/                        # Python cache (auto-created)
├── mlp_delta_e_model_lasso.h5          # Pre-trained Keras model
├── mlp_scaler_lasso.joblib             # Fitted scaler for input normalization
├── mlp_model_features_lasso.joblib     # Feature columns list
├── FCNN_Random.py                      # Random composition search script
├── FCNN_Spesific.py                    # Specific composition prediction script
└── FCNN_Model_Egitimi.ipynb           # Model training notebook
```

## 🔧 Model Details

- **Model Type**: Fully Connected Neural Network (MLP)
- **Framework**: TensorFlow/Keras
- **Port**: 5004
- **Python Version**: 3.7+ (3.13 supported)
- **Architecture**: Multi-layer perceptron with Lasso feature selection

### Dependencies
- `tensorflow>=2.13.0` - Deep learning framework
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Machine learning utilities
- `joblib>=1.0.0` - Model serialization
- `flask>=2.0.0` - Web API framework
- `flask-cors>=4.0.0` - Cross-origin resource sharing

## 🚀 Quick Setup

### Automated Setup
```bash
# Navigate to FCNN directory
cd lib/ml_models/fcnn

# Run setup script (creates venv and installs dependencies)
python setup.py

# Start the service (optional - app does this automatically)
python fcnn_service.py
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install tensorflow>=2.13.0 pandas numpy scikit-learn joblib flask flask-cors

# Start service
python fcnn_service.py
```

## 🔄 API Endpoints

### Health Check
```http
GET /health
Response: {"status": "healthy", "service": "FCNN ML Service", "model_loaded": true}
```

### Root Endpoint
```http
GET /
Response: {"status": "healthy", "service": "FCNN ML Service", "features_count": 123}
```

### Predict
```http
POST /predict
Content-Type: application/json

Request Body:
{
  "elements": ["Fe", "Ni", "Al"],
  "compositions": {"Fe": 0.5, "Ni": 0.3, "Al": 0.2},  // Optional
  "iterations": 1000                                   // For random search
}

Response:
{
  "delta_e": -2.345,
  "composition": {"Fe": 0.5, "Ni": 0.3, "Al": 0.2},
  "total": 1.0
}
```

## 🎯 Model Features

### Prediction Modes
1. **Fixed Composition**: Predict delta E for user-specified compositions
2. **Random Search**: Find optimal composition through random sampling

### Input Features
- **Element Fractions**: Composition percentages for each element
- **comp_ntypes**: Number of elements in the composition (auto-calculated)
- **Additional Features**: As defined in the feature selection process

### Model Architecture
- **Input Layer**: Feature vector based on Lasso feature selection
- **Hidden Layers**: Fully connected layers with activation functions
- **Output Layer**: Single neuron for delta E prediction
- **Preprocessing**: StandardScaler for input normalization

## 🔧 Technical Implementation

### Model Loading
```python
# Load pre-trained model
model = load_model('mlp_delta_e_model_lasso.h5', compile=False)

# Load preprocessing components
scaler = joblib.load('mlp_scaler_lasso.joblib')
feature_cols = joblib.load('mlp_model_features_lasso.joblib')
```

### Prediction Process
1. **Input Validation**: Check composition sums to 1.0
2. **Feature Engineering**: Create feature vector from composition
3. **Preprocessing**: Apply fitted scaler
4. **Prediction**: Forward pass through neural network
5. **Post-processing**: Extract and format results

### Random Search Algorithm
1. Generate random compositions with minimum fractions
2. Ensure composition constraints (sum = 1.0)
3. Predict delta E for each composition
4. Track best (minimum) delta E value
5. Return optimal composition and energy

## 🐛 Troubleshooting

### Common Issues

#### TensorFlow Installation
```bash
# For Python 3.9+, use standard TensorFlow
pip install tensorflow

# For compatibility issues
pip install tensorflow-cpu
```

#### Model Loading Errors
- Ensure all three files exist: `.h5`, scaler `.joblib`, features `.joblib`
- Check file permissions and corruption
- Verify TensorFlow version compatibility

#### Service Startup Issues
```bash
# Check port availability
netstat -ano | findstr :5004  # Windows
lsof -i :5004                 # macOS/Linux

# Kill conflicting processes if needed
taskkill /F /PID <PID>        # Windows
kill -9 <PID>                 # macOS/Linux
```

#### Memory Issues
- The FCNN model is lightweight but TensorFlow initialization requires ~500MB RAM
- Consider adjusting batch size for large prediction sets

## 📊 Model Performance

### Training Details
- **Feature Selection**: Lasso regularization for automatic feature selection
- **Validation**: Cross-validation during training
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and L2 regularization to prevent overfitting

### Expected Performance
- **Prediction Speed**: ~1-10ms per composition
- **Memory Usage**: ~500MB (TensorFlow overhead)
- **Accuracy**: Model-specific metrics available in training notebook

## 🔗 Integration Points

### Flutter Integration
- **Dart Model**: `lib/models/fcnn_model.dart`
- **Factory Registration**: Registered in `model_factory.dart`
- **Asset Management**: Model files included in `pubspec.yaml`

### Service Communication
- **Protocol**: HTTP REST API
- **Data Format**: JSON request/response
- **Error Handling**: Structured error responses
- **Monitoring**: Health check endpoints

## 📝 Development Notes

### Adding New Features
1. Retrain model with new features
2. Update feature list in `.joblib` file
3. Ensure scaler includes new features
4. Test prediction pipeline

### Model Updates
1. Replace `.h5` model file
2. Update corresponding scaler and features
3. Verify compatibility with service code
4. Test with sample predictions

### Performance Optimization
- Use `verbose=0` for silent predictions
- Batch predictions for multiple compositions
- Consider model quantization for production
- Monitor memory usage during extended use

## 📚 Related Scripts

- **FCNN_Random.py**: Standalone random search implementation
- **FCNN_Spesific.py**: Standalone specific composition prediction
- **FCNN_Model_Egitimi.ipynb**: Complete model training pipeline

For more details about the ML Models architecture, see the main [ML Models README](../README.md). 