import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY', '')

print("OpenAI API Key Check")
print("=" * 40)
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key)}")
print(f"API Key format: {api_key[:15]}...{api_key[-10:] if len(api_key) > 25 else ''}")
print(f"Starts with sk-proj: {'Yes' if api_key.startswith('sk-proj-') else 'No'}")

# Test imports first
print("\nTesting imports...")
try:
    from openai import OpenAI
    print("✅ OpenAI library imported successfully")
except ImportError as e:
    print(f"❌ Failed to import OpenAI: {e}")
    print("Try: pip install openai")
    sys.exit(1)

# Test API connection
if api_key:
    print("\nTesting API connection...")
    try:
        client = OpenAI(api_key=api_key)
        print("✅ Client created successfully")
        
        # Try a simple API call
        print("Fetching available models...")
        models = client.models.list()
        model_list = list(models)
        print(f"✅ API Connection: SUCCESS")
        print(f"Available models: {len(model_list)}")
        
        # Check if TTS models are available
        tts_models = [m.id for m in model_list if 'tts' in m.id]
        print(f"TTS models: {tts_models}")
        
        # Test TTS specifically
        print("\nTesting TTS functionality...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Testing OpenAI TTS functionality."
        )
        print("✅ TTS test successful!")
        
        print("\n🎉 All tests passed! OpenAI API is working correctly.")
        print("The 🚀 HD Video creation should work now.")
        
    except Exception as e:
        print(f"\n❌ API Test Failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        error_str = str(e).lower()
        if "authentication" in error_str or "api_key" in error_str:
            print("\n🔧 Authentication Issue:")
            print("- Check if your OpenAI API key is valid")
            print("- Make sure you have credits in your OpenAI account")
            print("- Verify the API key hasn't expired")
        elif "rate_limit" in error_str:
            print("\n🔧 Rate Limit:")
            print("- You've exceeded the API rate limit")
            print("- Try again in a few minutes")
        elif "network" in error_str or "connection" in error_str:
            print("\n🔧 Network Issue:")
            print("- Check your internet connection")
            print("- Try again in a moment")
        else:
            print(f"\n🔧 Unknown error: {e}")
else:
    print("\n❌ No API key found in .env file")

print("\nTest completed.")
