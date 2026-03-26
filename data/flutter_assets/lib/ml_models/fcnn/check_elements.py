#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check which elements are available in the FCNN model features.
"""

import joblib
import os

def check_available_elements():
    try:
        # Load feature columns
        features_path = 'mlp_model_features_lasso.joblib'
        if not os.path.exists(features_path):
            print(f"Features file not found: {features_path}")
            return
        
        feature_cols = joblib.load(features_path)
        print(f"Total features: {len(feature_cols)}")
        
        # Filter out only element symbols (not other features like comp_ntypes)
        element_symbols = [col for col in feature_cols if len(col) <= 3 and col.isalpha()]
        
        print(f"Available elements ({len(element_symbols)}):")
        print(sorted(element_symbols))
        
        # Check specifically for common elements
        test_elements = ['Fe', 'Ni', 'Al', 'Cr', 'Co', 'Ti', 'V', 'Mn', 'Cu', 'Zn']
        print(f"\nChecking common elements:")
        for elem in test_elements:
            available = elem in feature_cols
            print(f"  {elem}: {'✓' if available else '✗'}")
        
        return feature_cols
        
    except Exception as e:
        print(f"Error loading features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_available_elements() 