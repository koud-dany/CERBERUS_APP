#!/usr/bin/env python3
"""
Audio diagnostic script for manga recap videos
Run this to check if MP4 files have audio tracks
"""

import os
import glob
from moviepy.editor import VideoFileClip

def test_video_audio(video_path):
    """Test if a video file has audio"""
    try:
        print(f"\n=== Testing: {video_path} ===")
        
        if not os.path.exists(video_path):
            print("❌ File not found!")
            return False
            
        # Load video
        video = VideoFileClip(video_path)
        
        print(f"📹 Video duration: {video.duration:.2f} seconds")
        print(f"📹 Video FPS: {video.fps}")
        print(f"📹 Video size: {video.size}")
        
        # Check audio
        if video.audio is None:
            print("❌ NO AUDIO TRACK FOUND!")
            return False
        else:
            print(f"🔊 Audio duration: {video.audio.duration:.2f} seconds")
            print(f"🔊 Audio FPS: {video.audio.fps}")
            print("✅ AUDIO TRACK EXISTS!")
            
            # Check if audio has actual content (not silent)
            try:
                # Get a small sample of audio data
                audio_array = video.audio.to_soundarray(t_start=0, t_end=min(1, video.audio.duration))
                max_amplitude = abs(audio_array).max() if len(audio_array) > 0 else 0
                
                if max_amplitude > 0.001:  # Threshold for silence
                    print(f"🎵 Audio has content (max amplitude: {max_amplitude:.4f})")
                else:
                    print("⚠️  Audio track exists but appears to be silent")
                    
            except Exception as e:
                print(f"⚠️  Could not analyze audio content: {e}")
        
        video.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing video: {e}")
        return False

def main():
    """Main function to test all MP4 files in uploads"""
    print("🔍 Searching for MP4 files in manga uploads...")
    
    # Find all MP4 files
    upload_pattern = "D:/manga_uploads/**/*.mp4"
    mp4_files = glob.glob(upload_pattern, recursive=True)
    
    if not mp4_files:
        print("No MP4 files found in uploads directory")
        print("Checking current directory...")
        local_mp4s = glob.glob("*.mp4")
        mp4_files.extend(local_mp4s)
    
    if not mp4_files:
        print("❌ No MP4 files found to test!")
        return
    
    print(f"Found {len(mp4_files)} MP4 file(s)")
    
    # Test each file
    working_files = 0
    for mp4_file in mp4_files:
        if test_video_audio(mp4_file):
            working_files += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"Total files tested: {len(mp4_files)}")
    print(f"Files with audio: {working_files}")
    print(f"Files without audio: {len(mp4_files) - working_files}")

if __name__ == "__main__":
    main()
