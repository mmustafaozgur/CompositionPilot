# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import joblib
from tensorflow.keras.models import load_model
import random
import json

# Set UTF-8 encoding for console output on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    # Also set environment variables for proper encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Allow Unicode characters in JSON responses
CORS(app)

def safe_str(obj):
    """Convert any object to a safe ASCII string."""
    try:
        return str(obj).encode('ascii', errors='replace').decode('ascii')
    except:
        return repr(obj).encode('ascii', errors='replace').decode('ascii')

def safe_print(message):
    """Safe print function that handles Unicode characters on Windows."""
    try:
        # Convert to safe ASCII string first
        safe_message = safe_str(message)
        print(safe_message)
    except:
        # Ultimate fallback
        print("Error: Could not display message due to encoding issues")

safe_print("Starting FCNN Flask service...")
safe_print(f"Python version: {sys.version}")
safe_print(f"Working directory: {os.getcwd()}")

# Load the model and required files at startup
model_path = os.getenv('MODEL_PATH')
scaler_path = os.getenv('SCALER_PATH')
features_path = os.getenv('FEATURES_PATH')

safe_print(f"MODEL_PATH environment variable: {model_path}")
safe_print(f"SCALER_PATH environment variable: {scaler_path}")
safe_print(f"FEATURES_PATH environment variable: {features_path}")

# Use default paths if environment variables are not set
if not model_path:
    model_path = os.path.join(os.path.dirname(__file__), 'mlp_delta_e_model_lasso.h5')
if not scaler_path:
    scaler_path = os.path.join(os.path.dirname(__file__), 'mlp_scaler_lasso.joblib')
if not features_path:
    features_path = os.path.join(os.path.dirname(__file__), 'mlp_model_features_lasso.joblib')

try:
    safe_print(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)
    safe_print("FCNN model loaded successfully")

    safe_print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    safe_print("Scaler loaded successfully")

    safe_print(f"Loading feature columns from: {features_path}")
    feature_cols = joblib.load(features_path)
    safe_print(f"Feature columns loaded successfully: {len(feature_cols)} features")
    
except Exception as e:
    safe_print(f"Error loading model, scaler or feature columns: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'healthy',
        'service': 'FCNN ML Service',
        'model_loaded': model is not None,
        'features_count': len(feature_cols),
        'python_version': sys.version
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'FCNN ML Service',
        'model_loaded': model is not None,
        'features_count': len(feature_cols),
        'python_version': sys.version
    })

@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

def generate_random_composition(elements, min_frac=0.01):
    """Generate random composition for the given elements."""
    n = len(elements)
    if n * min_frac > 1:
        raise ValueError("Minimum fraction too high for number of elements!")
    
    fixed_total = n * min_frac
    remainder = 1 - fixed_total
    random_props = np.random.dirichlet(np.ones(n))
    composition = {elem: min_frac + r * remainder for elem, r in zip(elements, random_props)}
    return composition

def predict_with_fixed_composition(model, composition, feature_cols, scaler):
    """Predict delta_e for a fixed composition using FCNN model."""
    # Check composition sum
    total = sum(composition.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Compositions must sum to 1.0, got: {total:.6f}")

    # Create input features with all features initialized to 0
    input_features = {col: 0.0 for col in feature_cols}
    
    # Set element fractions
    num_elements_in_comp = 0
    for elem, frac in composition.items():
        if elem in input_features:
            input_features[elem] = frac
            if frac > 1e-6:
                num_elements_in_comp += 1

    # Set comp_ntypes if it's a feature
    if 'comp_ntypes' in input_features:
        input_features['comp_ntypes'] = num_elements_in_comp

    # Create DataFrame and scale
    df_input = pd.DataFrame([input_features], columns=feature_cols)
    df_input_scaled = scaler.transform(df_input)

    # Make prediction
    delta_e_pred = model.predict(df_input_scaled, verbose=0).flatten()
    return float(delta_e_pred[0])

def find_min_delta_e_with_random(model, elements, feature_cols, scaler, n_iter=1000, min_frac=0.01):
    """Find minimum delta_e using random composition search."""
    best_delta_e = float("inf")
    best_comp = None
    
    # Filter elements that are available in the model features
    available_elements = [elem for elem in elements if elem in feature_cols]
    if not available_elements:
        raise ValueError("None of the provided elements are available in the model")
    
    for i in range(n_iter):
        comp_dict = generate_random_composition(available_elements, min_frac)
        
        try:
            delta_e_pred = predict_with_fixed_composition(model, comp_dict, feature_cols, scaler)
            
            if delta_e_pred < best_delta_e:
                best_delta_e = delta_e_pred
                best_comp = comp_dict.copy()
        
        except Exception as e:
            safe_print(f"Error in iteration {i}: {e}")
            continue
    
    # Ensure all values are JSON serializable
    if best_comp:
        best_comp = {k: float(v) for k, v in best_comp.items()}

    return best_delta_e, best_comp

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        safe_print(f"Received request data: {data}")  # Debug logging
        
        elements = data.get('elements', [])
        compositions = data.get('compositions', {})
        iterations = data.get('iterations', 1000)
        
        safe_print(f"Parsed elements: {elements}")      # Debug logging
        safe_print(f"Parsed compositions: {compositions}")  # Debug logging 
        safe_print(f"Parsed iterations: {iterations}")   # Debug logging
        
        if not elements:
            safe_print("Error: No elements provided")
            return jsonify({'error': 'No elements provided'}), 400
        
        # Filter elements that are available in the model features
        available_elements = [elem for elem in elements if elem in feature_cols]
        if not available_elements:
            safe_print("Error: None of the provided elements are available in the model")
            return jsonify({'error': 'None of the provided elements are available in the model'}), 400
        
        # Determine mode more explicitly
        has_compositions = bool(compositions and len(compositions) > 0)
        safe_print(f"Mode detection: has_compositions = {has_compositions}")
        
        # If compositions are provided, use them directly (SPECIFIC MODE)
        if has_compositions:
            safe_print(f"Using SPECIFIC mode for elements: {elements} with compositions: {compositions}")
            
            # Validate that compositions are provided for all selected elements
            for element in elements:
                if element not in compositions:
                    safe_print(f"Error: Composition not provided for element: {element}")
                    return jsonify({'error': f'Composition not provided for element: {element}'}), 400
            
            # Validate that compositions sum to approximately 1.0
            total_composition = sum(compositions.values())
            if abs(total_composition - 1.0) > 0.001:
                safe_print(f"Error: Compositions must sum to 1.0, got: {total_composition}")
                return jsonify({'error': f'Compositions must sum to 1.0, got: {total_composition}'}), 400
            
            # Predict with the provided fixed composition
            delta_e = predict_with_fixed_composition(model, compositions, feature_cols, scaler)
            
            # Ensure all values are JSON serializable and properly encoded
            result = {
                'delta_e': float(delta_e),
                'composition': {str(k): float(v) for k, v in compositions.items()},
                'total': float(total_composition)
            }
            
            # Return JSON response with proper UTF-8 encoding
            return jsonify(result)
        else:
            # Fall back to random composition search (RANDOM MODE)
            safe_print(f"Using RANDOM mode for elements: {available_elements} with {iterations} iterations")
            
            best_delta_e, best_comp = find_min_delta_e_with_random(
                model=model,
                elements=available_elements,
                feature_cols=feature_cols,
                scaler=scaler,
                n_iter=iterations,
                min_frac=0.01
            )
            
            safe_print(f"Random search completed: best_delta_e = {best_delta_e}, best_comp = {best_comp}")
            
            # Ensure all values are JSON serializable and properly encoded
            result = {
                'delta_e': float(best_delta_e),
                'composition': {str(k): float(v) for k, v in best_comp.items()} if best_comp else {},
                'total': float(sum(best_comp.values())) if best_comp else 0.0
            }
            
            safe_print(f"Returning result: {result}")
            
            # Return JSON response with proper UTF-8 encoding
            return jsonify(result)
        
    except UnicodeEncodeError as e:
        safe_print(f"Unicode encoding error: {e}")
        import traceback
        try:
            traceback.print_exc()
        except:
            safe_print("Unicode error occurred in traceback")
        return jsonify({'error': 'Character encoding error occurred'}), 500
    except Exception as e:
        # Ensure the error message is ASCII-safe
        error_msg = safe_str(f"Error in predict endpoint: {e}")
        safe_print(error_msg)
        safe_print(f"Request data that caused error: {safe_str(request.json)}")
        
        import traceback
        try:
            traceback.print_exc()
        except:
            safe_print("Error occurred in traceback due to encoding")
        
        # Return ASCII-safe error message
        return jsonify({'error': safe_str(str(e))}), 500

if __name__ == '__main__':
    safe_print("="*50)
    safe_print("Starting FCNN Flask Service on port 5004")
    safe_print("Health check available at: http://localhost:5004/health")
    safe_print("Model endpoint available at: http://localhost:5004/predict")
    safe_print("Routes listing available at: http://localhost:5004/routes")
    safe_print("="*50)
    
    try:
        app.run(host='0.0.0.0', port=5004, debug=False)
    except Exception as e:
        safe_print(f"Failed to start Flask server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 