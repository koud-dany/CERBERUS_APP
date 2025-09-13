#!/usr/bin/env python3
"""
Simple test script to verify audio generation and video creation
"""
import os
import asyncio
from app import generate_tts_sync
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip

def test_audio_generation():
    """Test basic audio generation"""
    print("🎵 Testing audio generation...")
    
    test_text = "This is a test of the audio generation system."
    test_output = "test_audio.mp3"
    
    try:
        # Clean up any existing test file
        if os.path.exists(test_output):
            os.remove(test_output)
        
        # Generate audio
        result = generate_tts_sync(test_text, test_output)
        
        if result and os.path.exists(test_output):
            file_size = os.path.getsize(test_output)
            print(f"✅ Audio generated successfully: {test_output} ({file_size} bytes)")
            
            # Test loading with MoviePy
            try:
                audio_clip = AudioFileClip(test_output)
                print(f"✅ Audio loads in MoviePy: duration={audio_clip.duration:.2f}s, fps={audio_clip.fps}")
                audio_clip.close()
                return True
            except Exception as moviepy_error:
                print(f"❌ MoviePy failed to load audio: {moviepy_error}")
                return False
        else:
            print(f"❌ Audio generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_output):
            os.remove(test_output)

def test_video_with_audio():
    """Test creating a simple video with audio"""
    print("🎬 Testing video creation with audio...")
    
    # Create a simple test image (solid color)
    test_image = "test_image.jpg"
    test_audio = "test_audio.mp3"
    test_video = "test_video.mp4"
    
    try:
        # Clean up any existing files
        for file in [test_image, test_audio, test_video]:
            if os.path.exists(file):
                os.remove(file)
        
        # Create a simple test image
        from PIL import Image
        img = Image.new('RGB', (1280, 720), color='blue')
        img.save(test_image)
        print(f"✅ Created test image: {test_image}")
        
        # Generate test audio
        test_text = "This is a test video with audio narration."
        result = generate_tts_sync(test_text, test_audio)
        
        if not result or not os.path.exists(test_audio):
            print(f"❌ Failed to generate test audio")
            return False
        
        print(f"✅ Created test audio: {test_audio}")
        
        # Create video with audio
        print("🎬 Creating video with audio...")
        
        # Load clips
        audio_clip = AudioFileClip(test_audio)
        video_clip = ImageClip(test_image).set_duration(audio_clip.duration)
        video_clip = video_clip.set_audio(audio_clip)
        
        # Write video
        video_clip.write_videofile(
            test_video,
            fps=24,
            codec='libx264',
            audio_codec='aac'
        )
        
        # Verify result
        if os.path.exists(test_video):
            file_size = os.path.getsize(test_video)
            print(f"✅ Video created successfully: {test_video} ({file_size} bytes)")
            
            # Verify video has audio
            from moviepy.editor import VideoFileClip
            test_clip = VideoFileClip(test_video)
            has_audio = test_clip.audio is not None
            print(f"🎵 Video has audio: {has_audio}")
            if has_audio:
                print(f"🎵 Audio duration: {test_clip.audio.duration:.2f}s")
            test_clip.close()
            
            return has_audio
        else:
            print(f"❌ Video file not created")
            return False
            
    except Exception as e:
        print(f"❌ Video test failed: {e}")
        return False
    finally:
        # Clean up
        for file in [test_image, test_audio, test_video]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    print("🧪 Starting audio and video tests...")
    
    audio_ok = test_audio_generation()
    video_ok = test_video_with_audio()
    
    print("\n📊 Test Results:")
    print(f"Audio Generation: {'✅ PASS' if audio_ok else '❌ FAIL'}")
    print(f"Video with Audio: {'✅ PASS' if video_ok else '❌ FAIL'}")
    
    if audio_ok and video_ok:
        print("\n🎉 All tests passed! Audio should work in your videos.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
