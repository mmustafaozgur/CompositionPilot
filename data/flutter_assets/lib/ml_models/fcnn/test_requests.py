#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to debug FCNN service requests.
This will help identify why Flutter app gets HTTP 500 while terminal works.
"""

import requests
import json
import time

BASE_URL = 'http://localhost:5004'

def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f'{BASE_URL}/health')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_flutter_app_request():
    """Test request format sent by Flutter app (RANDOM mode)"""
    print("\n=== Testing Flutter App Request (RANDOM mode) ===")
    
    # This mimics what the Flutter app sends for random mode
    request_data = {
        'elements': ['Fe', 'Ni'],
        'compositions': {},  # Empty dict for random mode
        'iterations': 1000,
        'model_version': 'random',  # Extra field from Flutter
        'additional_params': {}     # Extra field from Flutter
    }
    
    print(f"Sending request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f'{BASE_URL}/predict',
            headers={'Content-Type': 'application/json'},
            json=request_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_terminal_working_request():
    """Test request format that works from terminal (RANDOM mode)"""
    print("\n=== Testing Terminal Working Request (RANDOM mode) ===")
    
    # This is probably what works from your terminal
    request_data = {
        'elements': ['Fe', 'Ni'],
        'iterations': 1000
        # No compositions field at all
    }
    
    print(f"Sending request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f'{BASE_URL}/predict',
            headers={'Content-Type': 'application/json'},
            json=request_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_specific_mode_request():
    """Test SPECIFIC mode request"""
    print("\n=== Testing SPECIFIC Mode Request ===")
    
    request_data = {
        'elements': ['Fe', 'Ni'],
        'compositions': {'Fe': 0.7, 'Ni': 0.3},
        'iterations': 1,
        'model_version': 'specific'
    }
    
    print(f"Sending request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f'{BASE_URL}/predict',
            headers={'Content-Type': 'application/json'},
            json=request_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def main():
    print("FCNN Service Request Testing")
    print("=" * 50)
    
    # Wait a moment for service to be ready
    print("Waiting for service to be ready...")
    time.sleep(3)
    
    # Test health first
    if not test_health():
        print("Service not healthy, exiting...")
        return
    
    # Test different request formats
    test_flutter_app_request()
    test_terminal_working_request()
    test_specific_mode_request()
    
    print("\n=== Testing Complete ===")

if __name__ == '__main__':
    main() 