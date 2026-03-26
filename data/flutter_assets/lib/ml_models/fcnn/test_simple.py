#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test of FCNN random prediction functions.
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def test_manual_prediction():
    """Test the FCNN prediction manually to isolate the issue."""
    try:
        print("Loading model components...")
        
        # Load model, scaler, and features
        model = load_model('mlp_delta_e_model_lasso.h5', compile=False)
        scaler = joblib.load('mlp_scaler_lasso.joblib')
        feature_cols = joblib.load('mlp_model_features_lasso.joblib')
        
        print(f"Model loaded: {model is not None}")
        print(f"Scaler loaded: {scaler is not None}")
        print(f"Features loaded: {len(feature_cols)} features")
        
        # Test generate_random_composition
        test_elements = ['Fe', 'Ni']
        print(f"\nTesting random composition for: {test_elements}")
        
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
        
        comp = generate_random_composition(test_elements)
        print(f"Generated composition: {comp}")
        print(f"Composition sum: {sum(comp.values())}")
        
        # Test predict_with_fixed_composition
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
        
        print(f"\nTesting prediction with composition...")
        delta_e = predict_with_fixed_composition(model, comp, feature_cols, scaler)
        print(f"Predicted delta_e: {delta_e}")
        
        # Test find_min_delta_e_with_random with just a few iterations
        def find_min_delta_e_with_random(model, elements, feature_cols, scaler, n_iter=5, min_frac=0.01):
            """Find minimum delta_e using random composition search."""
            best_delta_e = float("inf")
            best_comp = None
            
            # Filter elements that are available in the model features
            available_elements = [elem for elem in elements if elem in feature_cols]
            print(f"Available elements for search: {available_elements}")
            
            if not available_elements:
                raise ValueError("None of the provided elements are available in the model")
            
            for i in range(n_iter):
                print(f"Iteration {i+1}/{n_iter}")
                comp_dict = generate_random_composition(available_elements, min_frac)
                print(f"  Generated composition: {comp_dict}")
                
                try:
                    delta_e_pred = predict_with_fixed_composition(model, comp_dict, feature_cols, scaler)
                    print(f"  Predicted delta_e: {delta_e_pred}")
                    
                    if delta_e_pred < best_delta_e:
                        best_delta_e = delta_e_pred
                        best_comp = comp_dict.copy()
                        print(f"  New best: {best_delta_e}")
                
                except Exception as e:
                    print(f"  Error in iteration {i}: {e}")
                    continue
            
            # Ensure all values are JSON serializable
            if best_comp:
                best_comp = {k: float(v) for k, v in best_comp.items()}

            return best_delta_e, best_comp
        
        print(f"\nTesting random search (5 iterations)...")
        best_delta_e, best_comp = find_min_delta_e_with_random(model, test_elements, feature_cols, scaler, n_iter=5)
        print(f"Best delta_e: {best_delta_e}")
        print(f"Best composition: {best_comp}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"Error in manual test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_manual_prediction() 