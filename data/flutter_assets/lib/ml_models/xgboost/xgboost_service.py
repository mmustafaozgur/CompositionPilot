from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__)
CORS(app)

print("Starting XGBoost Flask service...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Load the model and feature columns at startup
model_path = os.getenv('MODEL_PATH')
columns_path = os.getenv('COLUMNS_PATH')

print(f"MODEL_PATH environment variable: {model_path}")
print(f"COLUMNS_PATH environment variable: {columns_path}")

if not model_path or not columns_path:
    print("Error: Required environment variables MODEL_PATH and COLUMNS_PATH must be set")
    sys.exit(1)

try:
    print(f"Loading model from: {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    print("Model loaded successfully")

    print(f"Loading columns from: {columns_path}")
    df = pd.read_csv(columns_path)
    feature_cols = df.columns
    print(f"Columns loaded successfully: {len(feature_cols)} features")
except Exception as e:
    print(f"Error loading model or columns: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'healthy',
        'service': 'XGBoost ML Service',
        'model_loaded': bst is not None,
        'features_count': len(feature_cols),
        'python_version': sys.version
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'XGBoost ML Service',
        'model_loaded': bst is not None,
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
    n = len(elements)
    if n * min_frac > 1:
        raise ValueError("Minimum fraction too high for number of elements!")
    
    fixed_total = n * min_frac
    remainder = 1 - fixed_total
    random_props = np.random.dirichlet(np.ones(n))
    composition = {elem: min_frac + r * remainder for elem, r in zip(elements, random_props)}
    return composition

def find_min_delta_e_with_random(model, elements, feature_cols, n_iter=1000, min_frac=0.05):
    best_delta_e = float("inf")
    best_comp = None
    for i in range(n_iter):
        comp_dict = generate_random_composition(elements, min_frac)
        row_dict = {col: 0.0 for col in feature_cols}
        for elem, frac in comp_dict.items():
            if elem in row_dict:
                row_dict[elem] = frac

        df_input = pd.DataFrame([row_dict])
        dmatrix_input = xgb.DMatrix(df_input)
        
        delta_e_pred = model.predict(dmatrix_input)[0]
        if delta_e_pred < best_delta_e:
            best_delta_e = delta_e_pred
            best_comp = comp_dict
        
        best_comp = {k: float(v) for k, v in best_comp.items()}

    return best_delta_e, best_comp

def predict_with_fixed_composition(model, composition, feature_cols):
    """Predict delta_e for a fixed composition"""
    # Create a row with all features initialized to 0
    row_dict = {col: 0.0 for col in feature_cols}
    
    # Set the composition values
    for elem, frac in composition.items():
        if elem in row_dict:
            row_dict[elem] = frac

    df_input = pd.DataFrame([row_dict])
    dmatrix_input = xgb.DMatrix(df_input)
    
    delta_e_pred = model.predict(dmatrix_input)[0]
    return float(delta_e_pred)

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
            delta_e = predict_with_fixed_composition(bst, compositions, feature_cols)
            
            return jsonify({
                'delta_e': delta_e,
                'composition': {k: float(v) for k, v in compositions.items()},
                'total': float(total_composition)
            })
        else:
            # Fall back to random composition search (legacy behavior)
            best_delta_e, best_comp = find_min_delta_e_with_random(
                model=bst,
                elements=elements,
                feature_cols=feature_cols,
                n_iter=iterations,
                min_frac=0.05
            )
            
            return jsonify({
                'delta_e': float(best_delta_e),
                'composition': best_comp,
                'total': float(sum(best_comp.values()))
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("Starting XGBoost Flask Service on port 5000")
    print("Health check available at: http://localhost:5000/health")
    print("Model endpoint available at: http://localhost:5000/predict")
    print("Routes listing available at: http://localhost:5000/routes")
    print("="*50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start Flask server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 