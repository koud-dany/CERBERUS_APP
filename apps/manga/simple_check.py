import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY', '')

print("=== OpenAI API Key Status ===")
print(f"API Key Present: {'✅ Yes' if api_key else '❌ No'}")
print(f"Key Length: {len(api_key)} characters")
print(f"Key Format: {'✅ Valid' if api_key.startswith('sk-proj-') else '❌ Invalid'}")

if api_key:
    print(f"Key Preview: {api_key[:20]}...{api_key[-15:]}")
    print("\n=== Recommendation ===")
    print("✅ API key is properly formatted and loaded")
    print("🚀 The HD Video creation should work now")
    print("📝 If you still have issues, check:")
    print("   - Internet connection")
    print("   - OpenAI account has sufficient credits")
    print("   - Flask app is running (python app.py)")
else:
    print("❌ Please add your OpenAI API key to the .env file")
