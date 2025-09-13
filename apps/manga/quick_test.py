import os
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

# Test API connection
if api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        print("\nTesting API connection...")
        models = client.models.list()
        model_list = list(models)
        print(f"✅ API Connection: SUCCESS")
        print(f"Available models: {len(model_list)}")
        
        # Check if TTS models are available
        tts_models = [m.id for m in model_list if 'tts' in m.id]
        print(f"TTS models available: {tts_models}")
        
        print("\n✅ OpenAI API is working correctly!")
        print("The 🚀 HD Video creation should work now.")
        
    except Exception as e:
        print(f"\n❌ API Test Failed: {e}")
        if "authentication" in str(e).lower():
            print("🔧 This looks like an authentication issue.")
            print("Please check if your OpenAI API key is valid and has credits.")
        elif "rate_limit" in str(e).lower():
            print("🔧 Rate limit exceeded. Try again in a moment.")
        else:
            print("🔧 Unknown error. Check your internet connection.")
else:
    print("\n❌ No API key found in .env file")
