import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_tts():
    """Test OpenAI TTS functionality"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            print("❌ No OpenAI API key found")
            return False
            
        client = OpenAI(api_key=api_key)
        
        print("Testing OpenAI TTS...")
        
        # Test TTS functionality
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",
            input="Hello, this is a test of OpenAI text-to-speech functionality for the manga video app."
        )
        
        # Save a small test file
        test_file = "test_tts_output.mp3"
        with open(test_file, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            file_size = os.path.getsize(test_file)
            print(f"✅ TTS test successful! Generated {file_size} bytes of audio")
            
            # Clean up test file
            os.remove(test_file)
            return True
        else:
            print("❌ TTS test failed - no audio generated")
            return False
            
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

# Run the test
if __name__ == "__main__":
    print("OpenAI TTS Test for Manga Video App")
    print("=" * 50)
    
    success = test_openai_tts()
    
    if success:
        print("\n🎉 OpenAI TTS is ready for HD Video creation!")
        print("The '🚀 Create HD Video with AI' button should now work properly.")
    else:
        print("\n⚠️ OpenAI TTS test failed.")
        print("Check your API key and internet connection.")
