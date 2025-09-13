#!/usr/bin/env python3
"""
Test script to check if the upload endpoint is working
"""
import requests
import os

# Test upload endpoint
url = "http://127.0.0.1:5000/manga/upload"

# Create a small test file
test_content = b"Hello, this is a test image file content"
files = {'files': ('test.png', test_content, 'image/png')}

try:
    print("Testing upload endpoint...")
    response = requests.post(url, files=files)
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    print(f"Response content: {response.text}")
    
    if response.headers.get('content-type', '').startswith('application/json'):
        try:
            json_data = response.json()
            print(f"JSON response: {json_data}")
        except:
            print("Failed to parse as JSON")
    
except Exception as e:
    print(f"Error: {e}")
