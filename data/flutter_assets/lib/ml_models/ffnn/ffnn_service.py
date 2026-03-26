from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

print("Starting FFNN Flask service...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Load the model and feature columns at startup
model_path = os.getenv('MODEL_PATH')
columns_path = os.getenv('COLUMNS_PATH')
# Remove dependency on large dataset
data_path = os.getenv('DATA_PATH')  # Keep for compatibility but won't use

print(f"MODEL_PATH environment variable: {model_path}")
print(f"COLUMNS_PATH environment variable: {columns_path}")

# Use pre-generated scaler and feature files
scaler_path = os.path.join(os.path.dirname(__file__), 'ffnn_scaler.joblib')
feature_cols_path = os.path.join(os.path.dirname(__file__), 'ffnn_feature_columns.joblib')

if not model_path or not columns_path:
    print("Error: Required environment variables MODEL_PATH and COLUMNS_PATH must be set")
    sys.exit(1)

try:
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)
    print("FFNN model loaded successfully")

    print(f"Loading pre-generated scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("Pre-fitted scaler loaded successfully")

    print(f"Loading feature columns from: {feature_cols_path}")
    feature_cols = joblib.load(feature_cols_path)
    print(f"Feature columns loaded successfully: {len(feature_cols)} features")
    
except Exception as e:
    print(f"Error loading model, scaler or feature columns: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'healthy',
        'service': 'FFNN ML Service',
        'model_loaded': model is not None,
        'features_count': len(feature_cols),
        'python_version': sys.version
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'FFNN ML Service',
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

def generate_random_composition(elements, min_frac=0.05):
    """Generate random composition for the given elements."""
    n = len(elements)
    if n * min_frac > 1:
        raise ValueError("Minimum fraction too high for number of elements!")
    
    fixed_total = n * min_frac
    remainder = 1 - fixed_total
    random_props = np.random.dirichlet(np.ones(n))
    composition = {elem: min_frac + r * remainder for elem, r in zip(elements, random_props)}
    return composition

def predict_delta_e_from_composition(comp_dict):
    """
    Predict delta_e for given composition dictionary.
    comp_dict: {'Fe':0.5, 'Ni':0.3, 'Al':0.2, ...} (should sum to 1.0)
    """
    # Check composition sum
    total = sum(comp_dict.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Compositions must sum to 1.0, got: {total:.6f}")

    # Create input vector
    x = np.zeros((1, len(feature_cols)), dtype=float)
    for i, feat in enumerate(feature_cols):
        if feat in comp_dict:
            x[0, i] = comp_dict[feat]

    # Scale input
    x_scaled = scaler.transform(x)

    # Make prediction
    delta_e_pred = model.predict(x_scaled)[0, 0]
    return float(delta_e_pred)

def find_min_delta_e_with_random(elements, n_iter=1000, min_frac=0.05):
    """Find minimum delta_e using random composition search."""
    best_delta_e = float("inf")
    best_comp = None
    
    for i in range(n_iter):
        comp_dict = generate_random_composition(elements, min_frac)
        
        try:
            delta_e_pred = predict_delta_e_from_composition(comp_dict)
            
            if delta_e_pred < best_delta_e:
                best_delta_e = delta_e_pred
                best_comp = comp_dict.copy()
        
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    # Ensure all values are JSON serializable
    if best_comp:
        best_comp = {k: float(v) for k, v in best_comp.items()}

    return best_delta_e, best_comp

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        elements = data.get('elements', [])
        compositions = data.get('compositions', {})
        iterations = data.get('iterations', 1000)
        
        if not elements:
            return jsonify({'error': 'No elements provided'}), 400
        
        # If compositions are provided, use them directly
        if compositions and len(compositions) > 0:
            # Validate that compositions are provided for all selected elements
            for element in elements:
                if element not in compositions:
                    return jsonify({'error': f'Composition not provided for element: {element}'}), 400
            
            # Validate that compositions sum to approximately 1.0
            total_composition = sum(compositions.values())
            if abs(total_composition - 1.0) > 0.001:
                return jsonify({'error': f'Compositions must sum to 1.0, got: {total_composition}'}), 400
            
            # Predict with the provided fixed composition
            delta_e = predict_delta_e_from_composition(compositions)
            
            return jsonify({
                'delta_e': delta_e,
                'composition': {k: float(v) for k, v in compositions.items()},
                'total': float(total_composition)
            })
        else:
            # Fall back to random composition search (legacy behavior)
            best_delta_e, best_comp = find_min_delta_e_with_random(
                elements=elements,
                n_iter=iterations,
                min_frac=0.05
            )
            
            return jsonify({
                'delta_e': float(best_delta_e),
                'composition': best_comp,
                'total': float(sum(best_comp.values()))
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("Starting FFNN Flask Service on port 5002")
    print("Health check available at: http://localhost:5002/health")
    print("Model endpoint available at: http://localhost:5002/predict")
    print("Routes listing available at: http://localhost:5002/routes")
    print("="*50)
    
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except Exception as e:
        print(f"Failed to start Flask server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 