import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test the API key
api_key = os.getenv('OPENAI_API_KEY', '')
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key starts with: {api_key[:15]}..." if api_key else "No API key")
print(f"API Key length: {len(api_key)}")

# Quick validation
if api_key.startswith('sk-'):
    print("✅ API key format looks correct")
else:
    print("❌ API key format looks incorrect")

try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Test with a simple API call
    models = client.models.list()
    print("✅ OpenAI connection successful!")
    
except Exception as e:
    print(f"❌ OpenAI test failed: {e}")
