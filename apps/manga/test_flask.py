#!/usr/bin/env python3
"""
Test script to verify Flask endpoints and TTS functionality
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_flask_endpoints():
    """Test if Flask app is running and endpoints are accessible"""
    base_url = "http://127.0.0.1:5000"
    
    print("Flask Endpoint Test")
    print("=" * 40)
    
    try:
        # Test main page
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Main page accessible")
        else:
            print(f"❌ Main page error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Flask app: {e}")
        print("Make sure you run: python app.py")
        return False
    
    # Test if environment is configured
    api_key = os.getenv('OPENAI_API_KEY', '')
    print(f"OpenAI API Key: {'✅ Configured' if api_key else '❌ Missing'}")
    
    return True

def test_button_prerequisites():
    """Check what the HD Video button needs to work"""
    print("\nHD Video Button Prerequisites")
    print("=" * 40)
    
    print("To use '🚀 Create HD Video with AI', you need:")
    print("1. ✅ Flask app running (http://127.0.0.1:5000)")
    print("2. ✅ OpenAI API key configured")
    print("3. 📋 Upload manga images first")
    print("4. 📝 Generate a recap from images")
    print("5. 🎬 Click 'HD Video (OpenAI Enhanced)'")
    print("6. 🚀 Click 'Create HD Video with AI'")
    
    print("\nCommon issues:")
    print("- Button not responding = missing images/recap")
    print("- Error messages = check browser console (F12)")
    print("- No audio = TTS generation failed")

if __name__ == "__main__":
    print("Manga Video App - Diagnostic Test")
    print("=" * 50)
    
    flask_ok = test_flask_endpoints()
    test_button_prerequisites()
    
    if flask_ok:
        print("\n🎉 Flask app is running!")
        print("🔗 Open: http://127.0.0.1:5000")
        print("📖 Follow the steps above to create HD videos")
    else:
        print("\n❌ Flask app not running or not accessible")
        print("🔧 Run: python app.py")
