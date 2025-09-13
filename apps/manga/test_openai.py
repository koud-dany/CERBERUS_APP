#!/usr/bin/env python3
"""
Test script to verify OpenAI API key and connection
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY', '')

print("OpenAI API Key Test")
print("=" * 50)
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key)}")
print(f"API Key starts with: {api_key[:15]}..." if api_key else "No API key")

if api_key:
    try:
        # Test the API key
        client = OpenAI(api_key=api_key)
        
        print("\nTesting API connection...")
        
        # Try a simple API call
        models = client.models.list()
        print("✅ API connection successful!")
        print(f"Available models: {len(list(models))}")
        
        # Test TTS specifically
        print("\nTesting TTS functionality...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Hello, this is a test."
        )
        print("✅ TTS test successful!")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        
        # Check if it's an API key issue
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            print("\n🔧 Possible solutions:")
            print("1. Check if your OpenAI API key is valid")
            print("2. Make sure you have credits in your OpenAI account")
            print("3. Verify the API key format (should start with 'sk-')")
else:
    print("❌ No OpenAI API key found in .env file")
