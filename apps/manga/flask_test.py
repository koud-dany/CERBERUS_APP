#!/usr/bin/env python3
"""
Simple Flask server test
"""
import os
import sys

print("Flask App Diagnostic Test")
print("=" * 50)

# Check if we're in the right directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Check if app.py exists
app_py_exists = os.path.exists('app.py')
print(f"app.py exists: {app_py_exists}")

# Check if .env exists
env_exists = os.path.exists('.env')
print(f".env exists: {env_exists}")

if env_exists:
    # Check .env contents
    with open('.env', 'r') as f:
        env_content = f.read()
    
    has_openai_key = 'OPENAI_API_KEY=' in env_content
    has_claude_key = 'CLAUDE_API_KEY=' in env_content
    
    print(f"OpenAI API key in .env: {has_openai_key}")
    print(f"Claude API key in .env: {has_claude_key}")

# Try to import dependencies
print("\nDependency Check:")
print("-" * 30)

try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask: {e}")

try:
    import openai
    print(f"✅ OpenAI: {openai.__version__}")
except ImportError as e:
    print(f"❌ OpenAI: {e}")

try:
    import anthropic
    print("✅ Anthropic: Available")
except ImportError as e:
    print(f"❌ Anthropic: {e}")

try:
    import moviepy
    print("✅ MoviePy: Available")
except ImportError as e:
    print(f"❌ MoviePy: {e}")

try:
    import edge_tts
    print("✅ Edge TTS: Available")
except ImportError as e:
    print(f"❌ Edge TTS: {e}")

print("\nIf all dependencies are available, you can start the Flask app with:")
print("python app.py")
