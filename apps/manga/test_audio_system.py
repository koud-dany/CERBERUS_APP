#!/usr/bin/env python3
"""
Audio Test Script - Run this to test if TTS and audio creation is working
"""

import requests
import json

def test_audio_system():
    """Test the audio creation system"""
    print("🎵 Testing Audio Creation System")
    print("=" * 50)
    
    try:
        # Test basic Flask connectivity
        response = requests.get('http://localhost:5000/debug-test', timeout=5)
        print("✅ Flask app is running")
        
        # Test audio creation
        print("\n🔊 Testing TTS audio creation...")
        audio_test_data = {
            'text': 'This is a test of the manga recap audio system. Testing OpenAI TTS, Windows Edge TTS, and silent audio fallback.'
        }
        
        response = requests.post(
            'http://localhost:5000/test-audio-creation',
            json=audio_test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Audio Test Results:")
            
            audio_results = result['audio_test_results']
            print(f"   OpenAI TTS: {'✅ Working' if audio_results['openai_tts'] else '❌ Failed'}")
            print(f"   Edge TTS: {'✅ Working' if audio_results['edge_tts'] else '❌ Failed'}")
            print(f"   Silent Audio: {'✅ Working' if audio_results['silent_audio'] else '❌ Failed'}")
            
            working_methods = result['summary']['working_methods']
            total_methods = result['summary']['total_methods']
            
            print(f"\n📈 Summary: {working_methods}/{total_methods} audio methods working")
            
            if audio_results.get('files_created'):
                print(f"\n📁 Files created during test:")
                for file_info in audio_results['files_created']:
                    print(f"   {file_info}")
            
            # Show any errors
            for key, value in audio_results.items():
                if key.endswith('_error'):
                    method = key.replace('_error', '').replace('_', ' ').title()
                    print(f"   ⚠️ {method} Error: {value}")
            
            if working_methods > 0:
                print(f"\n🎉 Audio system is working! ({working_methods} methods available)")
                print("✅ Video creation should include audio")
            else:
                print(f"\n❌ No audio methods working - videos will be silent")
                print("💡 Check your OpenAI API key and internet connection")
                
        else:
            print(f"❌ Audio test failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app")
        print("💡 Make sure to run: python app.py")
        print("💡 Then try this test again")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_audio_system()
