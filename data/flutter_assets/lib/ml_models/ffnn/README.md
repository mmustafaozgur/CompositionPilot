# FFNN Model (Feed-Forward Neural Network) 🧠

This directory contains the Feed-Forward Neural Network implementation for materials science prediction. It uses TensorFlow/Keras for deep learning and provides a Flask API for predictions.

## 🏗️ Architecture

The FFNN model consists of several interconnected components:

### Core Files
- **`ffnn_service.py`** - Main Flask API server (207 lines)
- **`setup.py`** - Automated setup with Python 3.13 compatibility (98 lines)
- **`nn_model.h5`** - Pre-trained TensorFlow/Keras model (517KB)
- **`ffnn_specific.py`** - Model-specific prediction logic (57 lines)
- **`ffnn_random.py`** - Random prediction fallback (93 lines)
- **`columns_data.csv`** - Feature column mappings (6KB)
- **`yeni_xgboost_veri.csv`** - Training dataset (784MB)

### Dependencies
- **TensorFlow**: `tf-nightly` (for Python 3.13 compatibility)
- **Protobuf**: `<5` (MessageFactory compatibility)
- **Flask & Flask-CORS**: Web API framework
- **Pandas, NumPy**: Data manipulation
- **Scikit-learn**: Additional ML utilities

## 🚀 Quick Start

### 1. Automatic Setup (Recommended)
```bash
cd lib/ml_models/ffnn
python setup.py
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install tf-nightly protobuf<5 flask flask-cors pandas numpy scikit-learn
```

### 3. Start the Service
```bash
# The Flutter app will start this automatically, but you can run manually:
python ffnn_service.py
# Service will be available at http://localhost:5001
```

## 🔧 Python 3.13 Compatibility

This model has special configuration for Python 3.13 support:

### TensorFlow Compatibility Issue
- **Problem**: Standard `tensorflow` package doesn't support Python 3.13
- **Solution**: Use `tf-nightly` (TensorFlow 2.20.0 dev build)
- **Alternative**: Downgrade to Python 3.11 or 3.12

### Protobuf Compatibility
- **Problem**: TensorFlow 2.20+ requires protobuf <5 for MessageFactory
- **Solution**: Automatically installs `protobuf<5`

### Automatic Detection
The setup script automatically detects Python version and installs appropriate packages:

```python
required_packages = [
    'flask',
    'flask-cors', 
    'tf-nightly',  # TensorFlow nightly for Python 3.13
    'protobuf<5',  # Compatible protobuf version
    'pandas',
    'numpy',
    'scikit-learn',
]
```

## 🌐 API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
    "status": "healthy",
    "model": "ffnn",
    "tensorflow_version": "2.20.0-dev20241204"
}
```

### Prediction
```http
POST /predict
Content-Type: application/json
```
**Request:**
```json
{
    "features": [0.1, 0.2, 0.3, ...],
    "iterations": 1000
}
```
**Response:**
```json
{
    "prediction": 123.45,
    "model": "ffnn",
    "method": "neural_network",
    "timestamp": "2024-01-01T12:00:00Z",
    "confidence": 0.87
}
```

## 🧠 Model Details

### Architecture
- **Type**: Feed-Forward Neural Network
- **Framework**: TensorFlow/Keras
- **Input Features**: Material composition features
- **Output**: Material property prediction
- **Training Data**: 784MB dataset (`yeni_xgboost_veri.csv`)

### Model File Structure
```
nn_model.h5
├── Architecture Definition
├── Weights and Biases  
├── Optimizer State
└── Training Configuration
```

### Prediction Pipeline
1. **Input Validation**: Check feature dimensions and types
2. **Preprocessing**: Normalize and scale input features
3. **Neural Network**: Forward pass through trained model
4. **Postprocessing**: Apply inverse transforms to output
5. **Confidence Estimation**: Calculate prediction confidence

## 🔍 Implementation Details

### Service Architecture
```python
# Main service file structure
ffnn_service.py
├── Flask App Initialization
├── CORS Configuration
├── Health Check Endpoint
├── Prediction Endpoint
├── Error Handling
└── Model Loading Logic
```

### Error Handling
The service includes robust error handling:
- TensorFlow import failures → Fallback to random predictions
- Model loading errors → Graceful degradation
- Invalid input data → Detailed error messages
- Memory issues → Automatic cleanup

### Fallback Mechanisms
1. **Primary**: TensorFlow neural network prediction
2. **Secondary**: Scikit-learn based prediction (`ffnn_specific.py`)
3. **Fallback**: Random prediction with realistic bounds (`ffnn_random.py`)

## 🐛 Troubleshooting

### Common Issues

#### TensorFlow Import Error
```
ImportError: No module named 'tensorflow'
```
**Solution:**
```bash
# For Python 3.13
pip install tf-nightly

# For Python 3.11/3.12
pip install tensorflow
```

#### Protobuf Version Conflict
```
AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
```
**Solution:**
```bash
pip install "protobuf<5"
```

#### Model Loading Error
```
OSError: Unable to open file (file signature not found)
```
**Solutions:**
- Verify `nn_model.h5` exists and isn't corrupted
- Re-download the model file
- Check file permissions

#### Memory Issues
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solutions:**
- Reduce batch size in predictions
- Close other memory-intensive applications
- Consider using GPU if available

#### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solutions:**
```bash
# Find process using port 5001
netstat -ano | findstr :5001  # Windows
lsof -i :5001                 # macOS/Linux

# Kill the process
taskkill /F /PID <PID>        # Windows  
kill -9 <PID>                 # macOS/Linux
```

### Debug Mode
Enable debug mode for detailed logging:
```python
# In ffnn_service.py
app.run(host='localhost', port=5001, debug=True)
```

## 🔧 Development

### Model Retraining
To retrain the model with new data:

1. **Prepare Data**: Update `yeni_xgboost_veri.csv`
2. **Train Model**: Use TensorFlow/Keras training script
3. **Export Model**: Save as `nn_model.h5`
4. **Update Columns**: Modify `columns_data.csv` if features change
5. **Test**: Verify predictions work correctly

### Adding Features
To add new input features:

1. **Update Dataset**: Add columns to training data
2. **Retrain Model**: Include new features in training
3. **Update Columns**: Modify `columns_data.csv`
4. **Update API**: Ensure feature validation handles new inputs

### Performance Optimization
- **Model Optimization**: Use TensorFlow Lite for faster inference
- **Caching**: Implement prediction caching for repeated inputs
- **Batch Processing**: Handle multiple predictions efficiently
- **GPU Acceleration**: Enable GPU support for faster computation

## 📊 Performance Metrics

### Typical Performance
- **Startup Time**: 15-30 seconds (model loading)
- **Prediction Time**: 10-50ms per prediction
- **Memory Usage**: 200-500MB
- **Model Accuracy**: Depends on training data quality

### Monitoring
The Flutter app monitors:
- Service health status
- Response times
- Error rates
- Memory usage

## 🔄 Version History

### v1.1.0 (Current)
- ✅ Python 3.13 compatibility with tf-nightly
- ✅ Stream controller lifecycle fixes
- ✅ Improved error handling and fallbacks
- ✅ Automatic dependency management

### v1.0.0 (Initial)
- ✅ Basic TensorFlow neural network
- ✅ Flask API implementation
- ✅ Integration with Flutter app

---

**For general setup instructions, see the main project README.** 