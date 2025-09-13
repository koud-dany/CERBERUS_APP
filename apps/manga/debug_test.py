#!/usr/bin/env python3
"""
Simple test to identify what's failing in the manga recap app
"""

import sys
import os
import requests
import time
import json
from datetime import datetime

def test_flask_app():
    """Test if the Flask app is running and responding"""
    print("🔍 Testing Flask App Connectivity")
    print("=" * 50)
    
    # Test if app is reachable
    try:
        response = requests.get('http://localhost:5000/debug-test', timeout=5)
        print(f"✅ Flask app responding: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Flask app not running - start it first!")
        print("   Run: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("⚠️ Flask app taking too long to respond")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Flask app: {e}")
        return False

def test_video_creation_endpoint():
    """Test the video creation endpoint with minimal data"""
    print("\n🎬 Testing Video Creation Endpoint")
    print("=" * 50)
    
    test_data = {
        'session_id': 'test_session_123',
        'recap': 'This is a short test recap for the manga video creation system. It should trigger the TTS pipeline.'
    }
    
    try:
        print("📡 Sending POST request to /create-openai-video...")
        response = requests.post(
            'http://localhost:5000/create-openai-video',
            json=test_data,
            timeout=30,
            stream=True
        )
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📡 Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ Error response: {response.text}")
            return False
            
        # Read streaming response
        print("📊 Reading streaming response...")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"   📦 Received: {decoded_line}")
                
                if 'error' in decoded_line.lower():
                    print(f"❌ Error detected in stream: {decoded_line}")
                    return False
                    
        print("✅ Video creation endpoint responding")
        return True
        
    except requests.exceptions.Timeout:
        print("⚠️ Video creation request timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing video creation: {e}")
        return False

def test_openai_api():
    """Test if OpenAI API is accessible"""
    print("\n🤖 Testing OpenAI API Access")
    print("=" * 50)
    
    try:
        from openai import OpenAI
        
        # Check if we can import the library
        print("✅ OpenAI library imported successfully")
        
        # Check for API key in environment
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✅ OpenAI API key found (length: {len(api_key)})")
            if api_key.startswith('sk-proj-sk-proj'):
                print("⚠️ API key has duplicate prefix - this might cause issues")
            elif api_key.startswith('sk-proj-'):
                print("✅ API key format looks correct")
            else:
                print("⚠️ API key format might be incorrect")
        else:
            print("❌ No OpenAI API key found in environment")
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OpenAI library: {e}")
        return False

def run_full_test():
    """Run all tests"""
    print("🚀 Manga Recap App Diagnostic Test")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: OpenAI API
    results['openai_api'] = test_openai_api()
    
    # Test 2: Flask connectivity
    results['flask_app'] = test_flask_app()
    
    # Test 3: Video creation (only if Flask is running)
    if results['flask_app']:
        results['video_creation'] = test_video_creation_endpoint()
    else:
        results['video_creation'] = False
        print("\n⏭️ Skipping video creation test - Flask app not running")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your app should be working correctly")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("   Check the error messages above for details")
        
        if not results['flask_app']:
            print("\n💡 NEXT STEPS:")
            print("   1. Start the Flask app: python app.py")
            print("   2. Wait for 'Running on http://127.0.0.1:5000'")
            print("   3. Run this test again")

if __name__ == "__main__":
    run_full_test()
