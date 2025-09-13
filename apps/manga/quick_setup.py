#!/usr/bin/env python3
"""
Quick setup script to identify and fix environment issues
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required keys"""
    print("🔍 Checking Environment Configuration")
    print("=" * 50)
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        print("❌ .env file not found")
        if env_example.exists():
            print("💡 Found .env.example file")
            print("   Please copy .env.example to .env and add your API keys")
        else:
            print("💡 Creating minimal .env file for testing...")
            create_minimal_env()
        return False
    else:
        print("✅ .env file found")
        
        # Check if it has the required keys
        with open('.env', 'r') as f:
            content = f.read()
            
        required_keys = ['CLAUDE_API_KEY', 'OPENAI_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if key not in content or f'{key}=' not in content:
                missing_keys.append(key)
                
        if missing_keys:
            print(f"⚠️ Missing keys in .env: {missing_keys}")
            return False
        else:
            print("✅ Required API keys found in .env")
            return True

def create_minimal_env():
    """Create a minimal .env file for testing"""
    env_content = """# Minimal .env file for testing
# Replace with your actual API keys for full functionality
CLAUDE_API_KEY=test_claude_key
OPENAI_API_KEY=test_openai_key
HEYGEN_API_KEY=test_heygen_key
FLASK_SECRET_KEY=test_secret_key_for_development_only
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created minimal .env file")
    print("⚠️ This uses test keys - add real API keys for full functionality")

def test_app_startup():
    """Test if the app can start with current configuration"""
    print("\n🚀 Testing App Startup")
    print("=" * 50)
    
    try:
        # Import Flask app to see if there are any import errors
        sys.path.insert(0, os.getcwd())
        
        print("📦 Importing Flask app...")
        from app import app
        print("✅ App imported successfully")
        
        print("🔧 Checking app configuration...")
        print(f"   Upload folder: {app.config.get('UPLOAD_FOLDER', 'Not set')}")
        print(f"   Claude API key: {'Set' if app.config.get('CLAUDE_API_KEY') else 'Not set'}")
        print(f"   OpenAI API key: {'Set' if app.config.get('OPENAI_API_KEY') else 'Not set'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ App configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function"""
    print("🛠️ Manga Recap App Quick Setup")
    print("=" * 50)
    
    # Check environment
    env_ok = check_env_file()
    
    # Test app startup
    app_ok = test_app_startup()
    
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    if env_ok and app_ok:
        print("🎉 SETUP COMPLETE!")
        print("✅ Environment configured")
        print("✅ App can start successfully")
        print("\n💡 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Upload manga images and test")
    else:
        print("⚠️ SETUP ISSUES DETECTED")
        if not env_ok:
            print("❌ Environment configuration needs attention")
            print("   Add your actual API keys to .env file")
        if not app_ok:
            print("❌ App startup issues detected")
            print("   Check error messages above")
        
        print("\n💡 To test with minimal setup:")
        print("   1. Use the test .env file created")
        print("   2. Run: python app.py")
        print("   3. Basic functions should work (except TTS)")

if __name__ == "__main__":
    main()
