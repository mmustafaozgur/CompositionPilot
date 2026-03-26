from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import joblib
import random

app = Flask(__name__)
CORS(app)

print("Starting CatBoost Flask service...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Load the model, scaler, and features at startup
model_path = os.path.join(os.path.dirname(__file__), 'catboost_delta_e_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'catboost_scaler.joblib')
features_path = os.path.join(os.path.dirname(__file__), 'catboost_model_features.joblib')

print(f"Model path: {model_path}")
print(f"Scaler path: {scaler_path}")
print(f"Features path: {features_path}")

try:
    print(f"Loading CatBoost model from: {model_path}")
    model = joblib.load(model_path)
    print("CatBoost model loaded successfully")

    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully")

    print(f"Loading features from: {features_path}")
    feature_cols = joblib.load(features_path)
    print(f"Features loaded successfully: {len(feature_cols)} features")
except Exception as e:
    print(f"Error loading model, scaler, or features: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'healthy',
        'service': 'CatBoost ML Service',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(feature_cols),
        'python_version': sys.version
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'CatBoost ML Service',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
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

def generate_random_composition(elements, min_frac=0.01, n_elements_limit=None):
    """
    Generate a random composition from the given elements list.
    Each element has a minimum fraction of min_frac and the total sums to 1.0.
    """
    if not elements:
        return {}

    if n_elements_limit is None or n_elements_limit > len(elements):
        n_elements_limit = len(elements)
    
    if n_elements_limit <= 0: 
        if elements: 
             num_selected_elements = random.randint(1, max(1, len(elements)))
        else: 
            return {}
    else: 
        num_selected_elements = random.randint(1, n_elements_limit)

    selected_elements = random.sample(elements, num_selected_elements)
    
    if not selected_elements:
        return {}

    fractions = np.random.dirichlet(np.ones(len(selected_elements)), size=1)[0]
    
    composition = {}
    adjusted_fractions = [max(frac, min_frac) for frac in fractions]
    
    total_adjusted = sum(adjusted_fractions)
    if total_adjusted == 0: 
        if len(selected_elements) > 0:
            equal_frac = 1.0 / len(selected_elements)
            for elem in selected_elements:
                composition[elem] = max(equal_frac, min_frac)
            current_sum_final = sum(composition.values())
            if current_sum_final > 0:
                for elem in composition:
                    composition[elem] /= current_sum_final
        return composition

    final_fractions = [f / total_adjusted for f in adjusted_fractions]

    for i, elem in enumerate(selected_elements):
        composition[elem] = final_fractions[i]
            
    current_sum_final = sum(composition.values())
    if current_sum_final > 0 and abs(current_sum_final - 1.0) > 1e-9: 
        for elem in composition:
            composition[elem] /= current_sum_final
                
    return composition

def find_min_delta_e_random(model, elements_to_vary, all_feature_cols, scaler,
                            n_iter=1000, min_frac=0.05, fixed_features=None):
    """
    Find the minimum Delta E by generating random compositions and predicting with CatBoost model.
    """
    min_delta_e = float('inf')
    best_composition_details = None
    
    if not elements_to_vary:
        print("Warning: `elements_to_vary` list is empty. Cannot perform random search.")
        return min_delta_e, best_composition_details

    print(f"Testing random compositions for {n_iter} iterations...")
    for i in range(n_iter):
        current_composition = generate_random_composition(elements_to_vary, min_frac=min_frac)
        if not current_composition:
            continue

        input_features = {col: 0.0 for col in all_feature_cols}
        num_elements_in_comp = 0
        for elem, frac in current_composition.items():
            if elem in input_features:
                input_features[elem] = frac
                if frac > 1e-6:
                    num_elements_in_comp += 1
        
        if fixed_features:
            for feat, val in fixed_features.items():
                if feat in input_features:
                    if val == "dynamic_from_comp" and feat == 'comp_ntypes':
                         input_features[feat] = num_elements_in_comp
                    else:
                        input_features[feat] = val
        
        if 'comp_ntypes' in input_features and \
           (not fixed_features or 'comp_ntypes' not in fixed_features or \
            (fixed_features and fixed_features.get('comp_ntypes') != "dynamic_from_comp")):
            input_features['comp_ntypes'] = num_elements_in_comp

        df_input = pd.DataFrame([input_features], columns=all_feature_cols)
        df_input_scaled = scaler.transform(df_input)
        
        try:
            predicted_delta_e_val = model.predict(df_input_scaled)
            current_delta_e = predicted_delta_e_val[0] if isinstance(predicted_delta_e_val, np.ndarray) and predicted_delta_e_val.ndim > 0 else predicted_delta_e_val
        except Exception as e:
            print(f"Prediction error: {e}, composition: {current_composition}")
            continue 

        if current_delta_e < min_delta_e:
            min_delta_e = current_delta_e
            best_composition_details = input_features.copy()
    
    # Extract the best composition for return
    best_comp = {}
    if best_composition_details:
        for elem in elements_to_vary:
            if elem in best_composition_details and best_composition_details[elem] > 1e-6:
                best_comp[elem] = best_composition_details[elem]
            
    return min_delta_e, best_comp

def predict_with_fixed_composition(model, composition, all_feature_cols, scaler, fixed_features=None):
    """Predict delta_e for a fixed composition using CatBoost model"""
    # Create input features with all features initialized to 0
    input_features = {col: 0.0 for col in all_feature_cols}
    
    num_elements_in_comp = 0
    for elem, frac in composition.items():
        if elem in input_features:
            input_features[elem] = frac
            if frac > 1e-6:
                num_elements_in_comp += 1
    
    # Set fixed features
    if fixed_features:
        for feat, val in fixed_features.items():
            if feat in input_features:
                if val == "dynamic_from_comp" and feat == 'comp_ntypes':
                    input_features[feat] = num_elements_in_comp
                else:
                    input_features[feat] = val
    
    # Set comp_ntypes if it's a feature
    if 'comp_ntypes' in input_features and \
       (not fixed_features or 'comp_ntypes' not in fixed_features or \
        (fixed_features and fixed_features.get('comp_ntypes') != "dynamic_from_comp")):
        input_features['comp_ntypes'] = num_elements_in_comp

    df_input = pd.DataFrame([input_features], columns=all_feature_cols)
    df_input_scaled = scaler.transform(df_input)
    
    predicted_delta_e_val = model.predict(df_input_scaled)
    delta_e = predicted_delta_e_val[0] if isinstance(predicted_delta_e_val, np.ndarray) and predicted_delta_e_val.ndim > 0 else predicted_delta_e_val
    
    return float(delta_e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        elements = data.get('elements', [])
        compositions = data.get('compositions', {})
        iterations = data.get('iterations', 1000)
        
        if not elements:
            return jsonify({'error': 'No elements provided'}), 400
        
        # Filter elements that are available in the model features
        available_elements = [elem for elem in elements if elem in feature_cols]
        if not available_elements:
            return jsonify({'error': 'None of the provided elements are available in the model'}), 400
        
        # Set up fixed features for CatBoost
        fixed_features = {}
        if 'comp_ntypes' in feature_cols:
            fixed_features['comp_ntypes'] = "dynamic_from_comp"
        
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
            
            # Filter compositions to only available elements
            filtered_compositions = {elem: frac for elem, frac in compositions.items() if elem in available_elements}
            
            if not filtered_compositions:
                return jsonify({'error': 'None of the provided compositions are for available elements'}), 400
            
            # Predict with the provided fixed composition
            delta_e = predict_with_fixed_composition(
                model, filtered_compositions, feature_cols, scaler, fixed_features
            )
            
            return jsonify({
                'delta_e': delta_e,
                'composition': {k: float(v) for k, v in compositions.items()},
                'total': float(total_composition)
            })
        else:
            # Fall back to random composition search (legacy behavior)
            best_delta_e, best_comp = find_min_delta_e_random(
                model=model,
                elements_to_vary=available_elements,
                all_feature_cols=feature_cols,
                scaler=scaler,
                n_iter=iterations,
                min_frac=0.05,
                fixed_features=fixed_features
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
    print("Starting CatBoost Flask Service on port 5001")
    print("Health check available at: http://localhost:5001/health")
    print("Model endpoint available at: http://localhost:5001/predict")
    print("Routes listing available at: http://localhost:5001/routes")
    print("Service ready!")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except Exception as e:
        print(f"Failed to start Flask server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 