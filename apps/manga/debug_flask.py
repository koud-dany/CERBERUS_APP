import requests
import json

# Test the Flask app endpoints
def test_flask_endpoints():
    base_url = "http://localhost:5000"
    
    print("Testing Flask app endpoints...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test 2: Main page
    try:
        response = requests.get(base_url)
        print(f"Main page: {response.status_code} - Content length: {len(response.text)}")
    except Exception as e:
        print(f"Main page failed: {e}")
    
    # Test 3: Test if OpenAI video endpoint accepts requests
    try:
        test_data = {
            "session_id": "test_session",
            "recap": "This is a test recap for debugging purposes."
        }
        
        response = requests.post(
            f"{base_url}/create-openai-video",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10  # Short timeout for debugging
        )
        print(f"OpenAI video endpoint: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
        else:
            print("Endpoint is responding correctly")
            
    except requests.exceptions.Timeout:
        print("OpenAI video endpoint: TIMEOUT - endpoint is hanging")
    except Exception as e:
        print(f"OpenAI video endpoint failed: {e}")

if __name__ == "__main__":
    test_flask_endpoints()
