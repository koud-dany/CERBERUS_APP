#!/usr/bin/env python3
"""
Test script for Windows Edge TTS functionality
"""

import asyncio
import edge_tts
import os

async def test_edge_tts():
    """Test Windows Edge TTS generation"""
    try:
        text = "Hello! This is a test of Windows Edge TTS for the manga recap application."
        output_path = "test_tts_output.mp3"
        voice = "en-US-JennyNeural"
        
        print(f"Testing Edge TTS...")
        print(f"Text: {text}")
        print(f"Voice: {voice}")
        print(f"Output: {output_path}")
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ SUCCESS: Audio file created!")
            print(f"File size: {file_size} bytes")
            
            # Clean up test file
            os.remove(output_path)
            print("Test file cleaned up.")
        else:
            print("❌ FAILED: Audio file was not created")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    print("=== Windows Edge TTS Test ===")
    asyncio.run(test_edge_tts())
    print("=== Test Complete ===")
