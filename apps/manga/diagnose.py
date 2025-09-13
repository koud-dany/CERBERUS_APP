#!/usr/bin/env python3
"""
Basic diagnostic for the manga app issue
"""

print("🔍 DIAGNOSING MANGA APP ISSUE")
print("=" * 50)

print("\n1. Testing Python basics...")
try:
    import sys
    print(f"✅ Python version: {sys.version}")
except Exception as e:
    print(f"❌ Python issue: {e}")

print("\n2. Testing Flask import...")
try:
    import flask
    print(f"✅ Flask version: {flask.__version__}")
except Exception as e:
    print(f"❌ Flask import failed: {e}")

print("\n3. Testing environment file...")
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_keys = {
        'OPENAI_API_KEY': len(os.getenv('OPENAI_API_KEY', '')) > 0,
        'CLAUDE_API_KEY': len(os.getenv('CLAUDE_API_KEY', '')) > 0,
    }
    
    for key, exists in api_keys.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {key}: {'Present' if exists else 'Missing'}")
        
except Exception as e:
    print(f"❌ Environment test failed: {e}")

print("\n4. Testing main app import...")
try:
    # Just test importing, not running
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    # Test individual imports that might be problematic
    print("   Testing moviepy...")
    import moviepy
    print("   ✅ moviepy imported")
    
    print("   Testing PIL...")
    from PIL import Image
    print("   ✅ PIL imported")
    
    print("   Testing openai...")
    from openai import OpenAI
    print("   ✅ OpenAI imported")
    
    print("   Testing edge_tts...")
    import edge_tts
    print("   ✅ edge_tts imported")
    
    print("   ✅ All dependencies imported successfully")
    
except Exception as e:
    print(f"❌ Import issue: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🎯 DIAGNOSIS COMPLETE")
print("\nIf all tests passed, the issue might be:")
print("1. App is starting but not showing output")
print("2. Port 5000 might be in use")
print("3. Windows firewall blocking the connection")
print("\n💡 Try opening: http://localhost:5000 in your browser")
print("   while running: python app.py")
