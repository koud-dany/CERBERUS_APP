from flask import Flask, request, render_template, jsonify, send_from_directory, Response
import os
import base64
import shutil
import traceback
import numpy as np
from PIL import Image, ImageFilter
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import edge_tts
from moviepy.editor import AudioFileClip

# Load environment variables from .env file
# Ensure we load from the app's directory, not the working directory
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Set temporary directory to avoid C: drive space issues
import tempfile
# Use D: drive for temporary files if available
if os.path.exists('D:\\'):
    temp_dir = 'D:\\temp_manga_app'
    os.makedirs(temp_dir, exist_ok=True)
    tempfile.tempdir = temp_dir
    print(f"Using temp directory: {temp_dir}")

import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from openai import OpenAI
import numpy as np
import requests
import time
import threading
import concurrent.futures
import tempfile
import zipfile
import re
import subprocess
import platform

# Fix for Pillow compatibility with MoviePy
# This addresses the "PIL.Image has no attribute 'ANTIALIAS'" error
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-flask-secret-key-here')
app.config['CLAUDE_API_KEY'] = os.getenv('CLAUDE_API_KEY', '')
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
app.config['HEYGEN_API_KEY'] = os.getenv('HEYGEN_API_KEY', '')
# Use D: drive for uploads to avoid C: drive space issues
if os.path.exists('D:\\'):
    app.config['UPLOAD_FOLDER'] = 'D:\\manga_uploads'
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max total upload size

# Add error handler for large uploads
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum total size is 100MB.'}), 413

# Add general error handler to ensure JSON responses
@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request. Please check your input.'}), 400

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename, session_id=None):
    """
    Create FFmpeg-safe filenames that avoid Windows path issues
    - Remove colons, semicolons, and other problematic characters
    - Use underscore separators instead of colons in timestamps
    - Ensure short, clean paths
    """
    # Remove or replace problematic characters
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    clean_name = re.sub(r'[\s\-\.]{2,}', '_', clean_name)  # Multiple spaces/dots/dashes to single underscore
    
    # If it's a timestamp-like pattern, clean it up
    clean_name = re.sub(r'(\d{8})[\-_:](\d{6})', r'\1_\2', clean_name)
    
    # Limit length and ensure extension
    name, ext = os.path.splitext(clean_name)
    if len(name) > 50:
        name = name[:50]
    
    return f"{name}{ext}"

def get_safe_temp_path(base_name="temp_audio"):
    """
    Create a safe temporary file path that won't conflict with FFmpeg
    Uses system temp directory with short, clean names
    """
    safe_name = sanitize_filename(base_name)
    temp_dir = tempfile.gettempdir()
    
    # Use timestamp with underscores only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_path = os.path.join(temp_dir, f"{safe_name}_{timestamp}.m4a")
    
    return safe_path

def check_ffmpeg_codecs():
    """
    Check which codecs are available in the FFmpeg installation
    Returns dict with codec availability
    """
    codecs = {
        'libx264': False,
        'aac': False,
        'h264': False,
        'mp3': False
    }
    
    try:
        # Run ffmpeg -codecs to check available codecs
        result = subprocess.run(['ffmpeg', '-codecs'], 
                              capture_output=True, text=True, timeout=10)
        output = result.stdout.lower()
        
        if 'libx264' in output:
            codecs['libx264'] = True
        if 'h264' in output and not codecs['libx264']:
            codecs['h264'] = True  # Fallback
        if 'aac' in output:
            codecs['aac'] = True
        if 'mp3' in output:
            codecs['mp3'] = True
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not check FFmpeg codecs: {e}")
        # Assume basic codecs are available
        codecs['libx264'] = True
        codecs['aac'] = True
    
    return codecs

async def generate_tts(text, output_path, voice="en-US-JennyNeural"):
    """
    Generate narration audio using Windows Edge TTS
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def generate_tts_sync(text, output_path, voice="en-US-JennyNeural"):
    """
    Synchronous wrapper for generate_tts using asyncio.run
    """
    try:
        asyncio.run(generate_tts(text, output_path, voice))
        return output_path
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None

def robust_write_videofile(video_clip, output_path, **kwargs):
    """
    Robust video writing with automatic fallback for FFmpeg issues
    Handles:
    - Windows path problems (safe filenames, no colons/spaces)
    - Missing codec support
    - Temp file cleanup issues
    - Memory optimization for large files
    - WebM fallback if MP4 fails
    """
    
    # Always use safe filename - no timestamps/colons that confuse FFmpeg
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    # Create safe MP4 filename
    safe_mp4_path = os.path.join(output_dir, "manga_video.mp4")
    
    # Check available codecs
    available_codecs = check_ffmpeg_codecs()
    
    # Determine best codecs based on availability
    video_codec = "libx264" if available_codecs['libx264'] else ("h264" if available_codecs['h264'] else "mpeg4")
    audio_codec = "aac" if available_codecs['aac'] else ("mp3" if available_codecs['mp3'] else "pcm_s16le")
    
    # Use safe temp audio path
    temp_audio_path = get_safe_temp_path("temp-audio.m4a") if audio_codec in ['aac', 'mp3'] else None
    
    # Base parameters with Windows-safe settings
    base_params = {
        'fps': kwargs.get('fps', 24),
        'codec': video_codec,
        'threads': 4,  # Limit threads to prevent "Broken pipe" errors
        'verbose': kwargs.get('verbose', False),
        'logger': kwargs.get('logger', None)
    }
    
    # Add audio parameters if video has audio
    if video_clip.audio is not None:
        base_params.update({
            'audio_codec': audio_codec,
            'temp_audiofile': temp_audio_path,
            'remove_temp': True
        })
    
    # Progressive quality attempts (high to low) with Windows-optimized FFmpeg params
    quality_configs = [
        # High quality attempt with Windows-safe parameters
        {
            **base_params,
            'bitrate': kwargs.get('bitrate', '3000k'),
            'ffmpeg_params': [
                '-movflags', '+faststart',  # Fix MP4 finalization issues
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'high',
                '-preset', 'medium',
                '-crf', '23',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts'
            ] + (['-shortest'] if video_clip.audio is not None else [])
        },
        # Medium quality fallback
        {
            **base_params,
            'bitrate': '2000k',
            'ffmpeg_params': [
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                '-crf', '28'
            ] + (['-shortest'] if video_clip.audio is not None else [])
        },
        # Minimal compatibility fallback
        {
            **base_params,
            'bitrate': '1000k',
            'ffmpeg_params': [
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        }
    ]
    
    # Try MP4 first with each quality config
    last_mp4_error = None
    for i, config in enumerate(quality_configs):
        try:
            print(f"Attempting MP4 write (quality level {i+1}/3)...")
            print(f"Using codec: {config['codec']}, audio: {config.get('audio_codec', 'none')}")
            
            # Clean up any existing temp files first
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
            
            video_clip.write_videofile(safe_mp4_path, **config)
            
            # Success! Clean up and return
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
            
            print(f"✅ MP4 video successfully written to: {safe_mp4_path}")
            return safe_mp4_path
            
        except Exception as e:
            last_mp4_error = e
            error_msg = str(e).lower()
            print(f"❌ MP4 quality level {i+1} failed: {e}")
            
            # Clean up failed temp files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
            
            # Check for specific FFmpeg errors and adjust next attempt
            if 'codec' in error_msg and i < len(quality_configs) - 1:
                print("Codec issue detected, trying compatibility fallback...")
                quality_configs[i+1]['codec'] = 'mpeg4'  # Most compatible
                if 'audio_codec' in quality_configs[i+1]:
                    quality_configs[i+1]['audio_codec'] = 'mp3'  # Most compatible
            
            continue
    
    # All MP4 attempts failed, try WebM fallback
    print(f"⚠️ MP4 export failed ({last_mp4_error}), retrying as WebM...")
    webm_path = os.path.join(output_dir, "manga_video.webm")
    
    try:
        webm_config = {
            'fps': kwargs.get('fps', 24),
            'codec': 'libvpx',
            'audio_codec': 'libvorbis' if video_clip.audio is not None else None,
            'threads': 4,
            'verbose': kwargs.get('verbose', False),
            'logger': kwargs.get('logger', None)
        }
        
        # Remove None values
        webm_config = {k: v for k, v in webm_config.items() if v is not None}
        
        video_clip.write_videofile(webm_path, **webm_config)
        print(f"✅ WebM video successfully written to: {webm_path}")
        return webm_path
        
    except Exception as webm_error:
        print(f"🚨 WebM fallback also failed: {webm_error}")
        raise Exception(f"Both MP4 and WebM encoding failed. MP4 error: {last_mp4_error}, WebM error: {webm_error}")

def memory_optimized_video_creation(clips, output_path, max_memory_mb=1024):
    """
    Create video with memory optimization for large files
    Processes clips in batches if needed
    """
    if not clips:
        raise ValueError("No clips provided for video creation")
    
    # Estimate memory usage (rough calculation)
    total_duration = sum(clip.duration for clip in clips)
    estimated_memory = total_duration * 50  # ~50MB per second estimate
    
    if estimated_memory > max_memory_mb:
        print(f"⚠️ Large video detected ({total_duration:.1f}s, ~{estimated_memory:.0f}MB)")
        print("Using memory-optimized batch processing...")
        
        # Process in smaller batches
        batch_size = max(1, int(max_memory_mb / 50))  # clips per batch
        batch_files = []
        
        try:
            for i in range(0, len(clips), batch_size):
                batch_clips = clips[i:i+batch_size]
                batch_output = f"{output_path}_batch_{i//batch_size}.mp4"
                
                print(f"Processing batch {i//batch_size + 1} ({len(batch_clips)} clips)...")
                from moviepy.editor import concatenate_videoclips
                batch_video = concatenate_videoclips(batch_clips, method="compose")
                
                robust_write_videofile(batch_video, batch_output)
                batch_files.append(batch_output)
                
                # Clean up batch clips from memory
                batch_video.close()
                for clip in batch_clips:
                    clip.close()
            
            # Combine all batches
            print("Combining batches into final video...")
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            
            batch_clips = [VideoFileClip(bf) for bf in batch_files]
            final_video = concatenate_videoclips(batch_clips, method="compose")
            
            result_path = robust_write_videofile(final_video, output_path)
            
            # Clean up
            final_video.close()
            for clip in batch_clips:
                clip.close()
            for bf in batch_files:
                try:
                    os.remove(bf)
                except:
                    pass
            
            return result_path
            
        except Exception as e:
            # Clean up batch files on error
            for bf in batch_files:
                try:
                    os.remove(bf)
                except:
                    pass
            raise e
    
    else:
        # Small enough for direct processing
        from moviepy.editor import concatenate_videoclips
        final_video = concatenate_videoclips(clips, method="compose")
        result_path = robust_write_videofile(final_video, output_path)
        final_video.close()
        return result_path

class VideoPipeline:
    """Automated video creation pipeline for manga recaps"""
    
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        self.fps = 24
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """Generate text embedding using OpenAI"""
        if not self.openai_client:
            print("No OpenAI client available, using fallback")
            return None
            
        try:
            response = self.openai_client.embeddings.create(
                model=model, 
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None  # Return None instead of random vector
    
    def get_batch_embeddings(self, texts, model="text-embedding-3-small"):
        """Generate embeddings for multiple texts in batch (more efficient)"""
        if not self.openai_client:
            print("No OpenAI client available, using fallback")
            return [None] * len(texts)
            
        try:
            # Limit batch size to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [np.array(emb.embedding) for emb in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return [None] * len(texts)
    
    def match_text_to_images(self, recap_lines, image_paths):
        """Match text lines to appropriate images using embeddings"""
        if not self.openai_client:
            # Simple fallback: cycle through images
            print("Using simple image cycling (no OpenAI client)")
            matched_images = []
            for i, line in enumerate(recap_lines):
                img_idx = i % len(image_paths)
                matched_images.append(image_paths[img_idx])
            return matched_images
        
        try:
            # Prepare texts for batch embedding
            valid_lines = [line for line in recap_lines if line.strip()]
            if not valid_lines:
                # Fallback to cycling if no valid text
                return [image_paths[i % len(image_paths)] for i in range(len(recap_lines))]
            
            # Generate embeddings for text segments (batch)
            line_embeddings = self.get_batch_embeddings([line[:500] for line in valid_lines])
            
            # Generate embeddings for images (batch)
            image_contexts = [f"manga panel {i+1} page {i+1} scene" for i in range(len(image_paths))]
            image_embeddings = self.get_batch_embeddings(image_contexts)
            
            # Check if any embeddings failed
            if any(emb is None for emb in line_embeddings + image_embeddings):
                print("Some embeddings failed, falling back to simple cycling")
                matched_images = []
                for i, line in enumerate(recap_lines):
                    img_idx = i % len(image_paths)
                    matched_images.append(image_paths[img_idx])
                return matched_images
            
            # Calculate similarities and match
            matched_images = []
            line_idx = 0
            
            for line in recap_lines:
                if not line.strip():
                    # For empty lines, use sequential mapping
                    img_idx = len(matched_images) % len(image_paths)
                    matched_images.append(image_paths[img_idx])
                    continue
                
                if line_idx >= len(line_embeddings):
                    # Fallback for extra lines
                    img_idx = len(matched_images) % len(image_paths)
                    matched_images.append(image_paths[img_idx])
                    continue
                
                line_emb = line_embeddings[line_idx]
                line_idx += 1
                
                # Calculate cosine similarity with all images
                similarities = []
                for img_emb in image_embeddings:
                    if np.linalg.norm(line_emb) > 0 and np.linalg.norm(img_emb) > 0:
                        sim = np.dot(line_emb, img_emb) / (np.linalg.norm(line_emb) * np.linalg.norm(img_emb))
                    else:
                        sim = 0
                    similarities.append(sim)
                
                best_idx = int(np.argmax(similarities))
                matched_images.append(image_paths[best_idx])
            
            return matched_images
            
        except Exception as e:
            print(f"Matching error: {e}")
            # Fallback to simple cycling
            matched_images = []
            for i, line in enumerate(recap_lines):
                img_idx = i % len(image_paths)
                matched_images.append(image_paths[img_idx])
            return matched_images
    
    def generate_tts(self, text, output_path):
        """Generate text-to-speech audio using Windows Edge TTS (reliable offline)"""
        try:
            print(f"Generating Windows TTS for text: {text[:50]}...")  # Debug log
            
            # Use Windows Edge TTS (offline, reliable)
            voice = "en-US-JennyNeural"  # High-quality Windows voice
            asyncio.run(generate_tts(text, output_path, voice))
            
            print(f"TTS file written successfully: {os.path.getsize(output_path)} bytes")  # Debug log
            return output_path
            
        except Exception as e:
            error_msg = str(e)
            print(f"Windows TTS generation error: {type(e).__name__}: {error_msg}")  # Debug log
            raise Exception(f"Windows TTS generation failed: {error_msg}")
    
    def split_text_into_segments(self, text, max_length=200):
        """Split long text into segments suitable for TTS and video timing"""
        sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_segment + " " + sentence) <= max_length:
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def create_video_package(self, session_folder, recap_text, output_name="manga_video"):
        """Create video creation package with TTS and matched images"""
        try:
            # Get image files in chronological order
            image_files = []
            for filename in sorted(os.listdir(session_folder)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    image_files.append(os.path.join(session_folder, filename))
            
            if not image_files:
                raise Exception("No image files found for video creation")
            
            # Split recap into segments for better timing
            text_segments = self.split_text_into_segments(recap_text)
            
            # Match text segments to images
            matched_images = self.match_text_to_images(text_segments, image_files)
            
            # Create video package folder
            video_package_folder = os.path.join(session_folder, 'automated_video_package')
            os.makedirs(video_package_folder, exist_ok=True)
            
            # Generate TTS for each segment (with Windows Edge TTS fallback)
            audio_files = []
            tts_failed = False
            
            for i, segment in enumerate(text_segments):
                audio_filename = f"segment_{i:03d}.mp3"
                audio_path = os.path.join(video_package_folder, audio_filename)
                
                audio_generated = False
                
                # Try OpenAI TTS first if available
                if self.openai_client:
                    try:
                        self.generate_tts(segment, audio_path)
                        audio_generated = True
                        print(f"✅ OpenAI TTS generated for segment {i}")
                    except Exception as e:
                        print(f"OpenAI TTS failed for segment {i}: {e}")
                
                # Fallback to Windows Edge TTS if OpenAI failed or not available
                if not audio_generated:
                    try:
                        # Use Windows Edge TTS as reliable fallback
                        result_path = generate_tts_sync(segment, audio_path, voice="en-US-JennyNeural")
                        if result_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            audio_generated = True
                            print(f"✅ Windows Edge TTS generated for segment {i}")
                        else:
                            print(f"❌ Windows Edge TTS failed for segment {i}")
                    except Exception as e:
                        print(f"❌ Windows Edge TTS error for segment {i}: {e}")
                
                # Create audio file mapping
                if audio_generated:
                    audio_files.append({
                        'audio_file': audio_filename,
                        'text': segment,
                        'image_file': os.path.basename(matched_images[i]),
                        'duration_estimate': len(segment) * 0.1  # rough estimate
                    })
                else:
                    print(f"⚠️ No audio generated for segment {i}, will create silent segment")
                    tts_failed = True
                    audio_files.append({
                        'audio_file': None,  # No audio file generated
                        'text': segment,
                        'image_file': os.path.basename(matched_images[i]),
                        'duration_estimate': len(segment) * 0.1  # rough estimate
                    })
            
            # Create video creation script (handles both with and without audio)
            script_content = self._generate_video_script(audio_files, text_segments, matched_images, tts_failed)
            
            with open(os.path.join(video_package_folder, 'video_creation_script.py'), 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Create segment mapping file
            mapping_content = "# AUDIO-IMAGE MAPPING\n\n"
            if tts_failed:
                mapping_content += "WARNING: TTS generation failed. You'll need to create audio files separately.\n\n"
            
            for i, item in enumerate(audio_files):
                mapping_content += f"Segment {i+1}:\n"
                if item['audio_file']:
                    mapping_content += f"  Audio: {item['audio_file']}\n"
                else:
                    mapping_content += f"  Audio: [NEEDS TO BE CREATED] - segment_{i:03d}.mp3\n"
                mapping_content += f"  Image: {item['image_file']}\n"
                mapping_content += f"  Text: {item['text'][:100]}...\n"
                mapping_content += f"  Estimated Duration: {item['duration_estimate']:.1f}s\n\n"
            
            with open(os.path.join(video_package_folder, 'segment_mapping.txt'), 'w', encoding='utf-8') as f:
                f.write(mapping_content)
            
            # Create TTS generation script if needed
            if tts_failed:
                tts_script = self._generate_tts_script(text_segments)
                with open(os.path.join(video_package_folder, 'generate_tts.py'), 'w', encoding='utf-8') as f:
                    f.write(tts_script)
            
            # Create usage instructions
            instructions = f"""# AUTOMATED VIDEO CREATION PACKAGE

This package contains everything needed to create your manga recap video.

## CONTENTS:
- segment_mapping.txt: Shows which text goes with which image
- video_creation_script.py: Python script to create the final video
- usage_instructions.txt: This file
{"- generate_tts.py: Script to generate audio files (TTS failed)" if tts_failed else "- segment_XXX.mp3: Audio files for each text segment"}

## STATUS:
{"WARNING: TTS GENERATION FAILED - You need to create audio files separately" if tts_failed else "SUCCESS: TTS files generated successfully"}

## REQUIREMENTS TO RUN THE VIDEO SCRIPT:
pip install moviepy pillow numpy

## NOTE ABOUT SUBTITLES:
Videos are created WITHOUT subtitles to avoid ImageMagick dependency issues.
To add subtitles, you can:
1. Install ImageMagick: https://imagemagick.org/script/download.php#windows
2. Configure MoviePy to find ImageMagick (see MoviePy docs)
3. Or add subtitles manually in video editing software

## IF TTS FAILED - CREATE AUDIO FILES:
1. Use the generate_tts.py script with a working OpenAI API key
2. Or use alternative TTS services:
   - ElevenLabs: https://elevenlabs.io/
   - Murf: https://murf.ai/
   - Google Cloud TTS
   - Azure Speech Services
3. Save audio files as segment_000.mp3, segment_001.mp3, etc.

## TO CREATE THE VIDEO:
1. Make sure all your manga images are in the parent folder
2. {"Create audio files using generate_tts.py or alternative TTS" if tts_failed else "Audio files are ready"}
3. Install the required packages above
4. Run: python video_creation_script.py

## MANUAL VIDEO EDITING:
You can also use the text segments and mapping manually in:
- Adobe Premiere Pro
- DaVinci Resolve
- CapCut
- Any video editing software

## TIPS:
- Each text segment is designed to match with a specific image
- The script will automatically create transitions and timing
- You can adjust timing and effects in the generated script
- Total estimated video length: {sum(item['duration_estimate'] for item in audio_files):.1f} seconds
"""
            
            with open(os.path.join(video_package_folder, 'usage_instructions.txt'), 'w', encoding='utf-8') as f:
                f.write(instructions)
            
            return {
                'package_folder': video_package_folder,
                'audio_files': len([f for f in audio_files if f['audio_file']]),
                'total_segments': len(text_segments),
                'estimated_duration': sum(item['duration_estimate'] for item in audio_files),
                'tts_failed': tts_failed,
                'audio_success_count': len([f for f in audio_files if f['audio_file']]),
                'tts_method': 'Windows Edge TTS' if not tts_failed else 'Failed - needs manual generation'
            }
            
        except Exception as e:
            raise Exception(f"Video package creation failed: {str(e)}")
    
    def _generate_video_script(self, audio_files, text_segments, matched_images, tts_failed=False):
        """Generate Python script for video creation"""
        if tts_failed:
            audio_check = """
        # Check if audio file exists
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found!")
            print("Please generate audio files first using generate_tts.py or alternative TTS service")
            print("Expected audio files: segment_000.mp3, segment_001.mp3, etc.")
            continue"""
        else:
            audio_check = """
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found, skipping...")
            continue"""
        
        return f'''#!/usr/bin/env python3
"""
Automated Video Creation Script
Generated by Manga Recap App
"""

import os
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx import all as vfx
from PIL import Image, ImageFilter

# Fix for Pillow compatibility with MoviePy
# This addresses the "PIL.Image has no attribute 'ANTIALIAS'" error
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

def create_video():
    """Create the final manga recap video"""
    clips = []
    
    # Video settings
    target_height = 720
    fps = 24
    
    segments = {audio_files}
    
    {"# NOTE: TTS generation failed. Audio files need to be created separately." if tts_failed else "# All audio files were generated successfully"}
    
    for i, segment in enumerate(segments):
        audio_file = segment['audio_file'] or f"segment_{{i:03d}}.mp3"
        if not os.path.exists(audio_file):
            print(f"⚠️ Missing {{audio_file}}, skipping")
            continue
        
        audio = AudioFileClip(audio_file)
        duration = audio.duration + 0.5
        
        # Precompute blurred background
        blurred_path = f"temp_blur_{{i}}.jpg"
        if not os.path.exists(blurred_path):
            img = Image.open(os.path.join("..", segment['image_file']))
            img.filter(ImageFilter.GaussianBlur(15)).save(blurred_path, "JPEG")

        background_clip = ImageClip(blurred_path).set_duration(duration).resize((1280,720))
        main_img_clip = ImageClip(os.path.join("..", segment['image_file'])).set_duration(duration).resize(height=720)
        main_img_clip = main_img_clip.on_color(size=(1280,720), color=(0,0,0), pos=("center","center"))

        img_clip = CompositeVideoClip([background_clip, main_img_clip])
        video_clip = img_clip.set_audio(audio)
        clips.append(video_clip)
    
    if not clips:
        print("Error: No valid clips created")
        {"print('Make sure audio files exist: segment_000.mp3, segment_001.mp3, etc.')" if tts_failed else ""}
        return
    
    print("Combining all clips...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Always write to a safe filename (no colons/spaces)
    output_file = "manga_video.mp4"
    print(f"Writing video to {{output_file}}...")
    
    # Use robust video writing with Windows-safe FFmpeg handling
    try:
        result_path = robust_write_videofile(
            final_video,
            output_file,
            fps=fps,
            verbose=True,
            logger='bar'
        )
        print(f"✅ Video creation complete! Output: {{result_path}}")
    except Exception as e:
        print(f"🚨 All video encoding attempts failed: {{e}}")
        print("Please check FFmpeg installation and try again.")
        return
    
    print(f"Video creation complete! Output: {{output_file}}")
    print(f"Total duration: {{final_video.duration:.1f}} seconds")

if __name__ == "__main__":
    create_video()
'''

    def _generate_tts_script(self, text_segments):
        """Generate a script to create TTS files separately"""
        return f'''#!/usr/bin/env python3
"""
TTS Generation Script
Use this to generate audio files with Windows TTS or alternative services
"""

import asyncio
import edge_tts
from openai import OpenAI
import os

# Text segments to convert to speech
text_segments = {text_segments}

async def generate_tts_windows(text, output_path, voice="en-US-JennyNeural"):
    """Generate narration audio using Windows Edge TTS"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def generate_with_windows_tts():
    """Generate TTS using Windows Edge TTS (recommended - offline, reliable)"""
    print("Generating audio with Windows Edge TTS...")
    
    for i, text in enumerate(text_segments):
        audio_filename = f"segment_{{i:03d}}.mp3"
        print(f"Generating {{audio_filename}}...")
        
        try:
            # Available voices: en-US-JennyNeural, en-US-AriaNeural, en-US-GuyNeural, etc.
            voice = "en-US-JennyNeural"  # High-quality female voice
            asyncio.run(generate_tts_windows(text, audio_filename, voice))
            print(f"SUCCESS: Created {{audio_filename}}")
            print(f"File size: {{os.path.getsize(audio_filename)}} bytes")
            
        except Exception as e:
            print(f"FAILED: Failed to generate {{audio_filename}}: {{e}}")

def generate_with_openai(api_key):
    """Generate TTS using OpenAI (requires API key and credits)"""
    client = OpenAI(api_key=api_key)
    
    for i, text in enumerate(text_segments):
        audio_filename = f"segment_{{i:03d}}.mp3"
        print(f"Generating {{audio_filename}}...")
        
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text[:4000]
            )
            
            with open(audio_filename, "wb") as f:
                f.write(response.read())  # Use .read() for new SDK
            print(f"SUCCESS: Created {{audio_filename}}")
            print(f"TTS file size: {{os.path.getsize(audio_filename)}} bytes")  # Debug file size
            
        except Exception as e:
            print(f"FAILED: Failed to generate {{audio_filename}}: {{e}}")

def generate_with_elevenlabs(api_key):
    """Generate TTS using ElevenLabs (requires elevenlabs package)"""
    try:
        from elevenlabs import generate, save
    except ImportError:
        print("ElevenLabs package not installed. Run: pip install elevenlabs")
        return
    
    for i, text in enumerate(text_segments):
        audio_filename = f"segment_{{i:03d}}.mp3"
        print(f"Generating {{audio_filename}} with ElevenLabs...")
        
        try:
            audio = generate(
                text=text,
                voice="Bella",  # or another voice
                api_key=api_key
            )
            save(audio, audio_filename)
            print(f"SUCCESS: Created {{audio_filename}}")
            
        except Exception as e:
            print(f"FAILED: Failed to generate {{audio_filename}}: {{e}}")

def main():
    print("TTS Generation Options:")
    print("1. Windows Edge TTS (Recommended - Offline, Reliable)")
    print("2. OpenAI TTS (Requires API key)")
    print("3. ElevenLabs (Requires API key)")
    print("4. Manual (create your own audio files)")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        generate_with_windows_tts()
    
    elif choice == "2":
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            generate_with_openai(api_key)
        else:
            print("No API key provided")
    
    elif choice == "3":
        api_key = input("Enter your ElevenLabs API key: ").strip()
        if api_key:
            generate_with_elevenlabs(api_key)
        else:
            print("No API key provided")
    
    elif choice == "4":
        print("Manual audio creation:")
        print("Create audio files with names:")
        for i in range(len(text_segments)):
            print(f"  segment_{{i:03d}}.mp3")
        print("\\nText for each segment:")
        for i, text in enumerate(text_segments):
            print(f"\\nSegment {{i:03d}}:")
            print(f"{{text}}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
'''

class HeyGenVideoAPI:
    """
    HeyGen API integration for AI video generation
    Provides professional AI avatar and video generation capabilities
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.heygen.com/v1"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'X-API-Key': api_key,
                'Content-Type': 'application/json',
                'User-Agent': 'MangaRecapApp/1.0'
            })
    
    def check_credits(self):
        """Check remaining API credits"""
        try:
            response = self.session.get(f"{self.base_url}/user/remaining_quota")
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'credits': data.get('credit_left', 0),
                    'subscription': data.get('subscription_type', 'free')
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_video_from_image(self, image_path, text, duration=4, avatar_id="josh_lite3_20230714"):
        """
        Generate video from manga image with AI avatar narration using HeyGen
        
        Args:
            image_path: Path to manga page image
            text: Text for AI avatar to speak
            duration: Video length preference (HeyGen auto-adjusts based on text)
            avatar_id: HeyGen avatar ID for narration
        """
        try:
            # Create video generation request with text-to-avatar
            generation_data = {
                "video_inputs": [
                    {
                        "character": {
                            "type": "avatar",
                            "avatar_id": avatar_id,
                            "avatar_style": "normal"
                        },
                        "voice": {
                            "type": "text",
                            "input_text": text,
                            "voice_id": "1bd001e7e50f421d891986aad5158bc8",  # Default English voice
                            "speed": 1.0
                        },
                        "background": {
                            "type": "image",
                            "url": self._upload_image(image_path)
                        }
                    }
                ],
                "aspect_ratio": "16:9",
                "test": False
            }
            
            response = self.session.post(f"{self.base_url}/video/generate", json=generation_data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'video_id': result.get('video_id'),
                    'estimated_time': 60,  # HeyGen typically takes 1-2 minutes
                    'cost_credits': 1
                }
            else:
                return {'success': False, 'error': f'Generation failed: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _upload_image(self, image_path):
        """Upload image to HeyGen and return URL"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                headers = {'X-API-Key': self.api_key}
                
                response = requests.post(f"{self.base_url}/asset/upload", 
                                       headers=headers, files=files)
                
                if response.status_code == 200:
                    return response.json().get('url')
                else:
                    raise Exception(f"Image upload failed: {response.text}")
        except Exception as e:
            raise Exception(f"Image upload error: {str(e)}")
    
    def check_generation_status(self, video_id):
        """Check the status of a video generation job"""
        try:
            response = self.session.get(f"{self.base_url}/video/{video_id}")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'processing').lower()
                
                # Map HeyGen status to our format
                if status == 'completed':
                    return {
                        'success': True,
                        'status': 'completed',
                        'progress': 100,
                        'video_url': data.get('video_url'),
                        'thumbnail_url': data.get('thumbnail_url'),
                        'error_message': None
                    }
                elif status == 'failed':
                    return {
                        'success': True,
                        'status': 'failed',
                        'progress': 0,
                        'video_url': None,
                        'error_message': data.get('error', 'Video generation failed')
                    }
                else:
                    return {
                        'success': True,
                        'status': 'processing',
                        'progress': data.get('progress', 50),
                        'video_url': None,
                        'error_message': None
                    }
            else:
                return {'success': False, 'error': f'Status check failed: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def download_video(self, video_url, output_path):
        """Download generated video from HeyGen"""
        try:
            response = self.session.get(video_url, stream=True)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return {'success': True, 'file_path': output_path}
            else:
                return {'success': False, 'error': f'Download failed: {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_manga_video_sequence(self, image_files, text_segments, output_folder):
        """
        Create a sequence of AI avatar videos from manga pages
        
        Args:
            image_files: List of manga image file paths
            text_segments: List of text descriptions for each image
            output_folder: Folder to save generated videos
        """
        if not self.api_key:
            return {'success': False, 'error': 'HeyGen API key required'}
        
        # Check credits first
        credits_check = self.check_credits()
        if not credits_check['success']:
            return {'success': False, 'error': f'Credits check failed: {credits_check["error"]}'}
        
        if credits_check['credits'] < len(image_files):
            return {
                'success': False, 
                'error': f'Insufficient credits. Need {len(image_files)}, have {credits_check["credits"]}'
            }
        
        results = []
        job_ids = []
        
        # Submit all generation jobs
        for i, (image_file, text) in enumerate(zip(image_files, text_segments)):
            # Create professional prompt for AI avatar
            enhanced_text = f"This is page {i+1} of a manga story. {text}"
            
            result = self.create_video_from_image(
                image_path=image_file,
                text=enhanced_text,
                duration=min(8, max(4, len(text) * 0.15)),  # Duration based on text length
                avatar_id="josh_lite3_20230714"  # Professional avatar
            )
            
            if result['success']:
                job_ids.append({
                    'job_id': result['job_id'],
                    'image_file': image_file,
                    'text': text,
                    'index': i
                })
                results.append({
                    'index': i,
                    'status': 'submitted',
                    'job_id': result['job_id']
                })
            else:
                results.append({
                    'index': i,
                    'status': 'failed',
                    'error': result['error']
                })
        
        return {
            'success': True,
            'job_ids': job_ids,
            'results': results,
            'total_jobs': len(job_ids),
            'estimated_time': max([job.get('estimated_time', 60) for job in job_ids]) if job_ids else 0
        }
    
    def wait_for_completion(self, job_ids, max_wait_time=600, callback=None):
        """
        Wait for all video generation jobs to complete
        
        Args:
            job_ids: List of job dictionaries from create_manga_video_sequence
            max_wait_time: Maximum time to wait in seconds
            callback: Optional callback function for progress updates
        """
        start_time = time.time()
        completed_videos = []
        
        while job_ids and (time.time() - start_time) < max_wait_time:
            for job in job_ids.copy():
                status = self.check_generation_status(job['job_id'])
                
                if status['success']:
                    if status['status'] == 'completed' and status['video_url']:
                        # Download the completed video
                        output_filename = f"segment_{job['index']:03d}_generated.mp4"
                        output_path = os.path.join(os.path.dirname(job['image_file']), output_filename)
                        
                        download_result = self.download_video(status['video_url'], output_path)
                        
                        if download_result['success']:
                            completed_videos.append({
                                'index': job['index'],
                                'video_path': output_path,
                                'image_file': job['image_file'],
                                'text': job['text']
                            })
                            job_ids.remove(job)
                            
                            if callback:
                                callback(f"✅ Video {job['index']+1} completed: {output_filename}")
                    
                    elif status['status'] == 'failed':
                        if callback:
                            callback(f"❌ Video {job['index']+1} failed: {status.get('error_message', 'Unknown error')}")
                        job_ids.remove(job)
                    
                    else:
                        # Still processing
                        if callback:
                            callback(f"🔄 Video {job['index']+1}: {status['status']} ({status['progress']}%)")
            
            if job_ids:  # Still waiting for some jobs
                time.sleep(10)  # Wait 10 seconds before checking again
        
        # Handle any remaining incomplete jobs
        for job in job_ids:
            if callback:
                callback(f"⏱️ Video {job['index']+1} timed out")
        
        return {
            'success': True,
            'completed_videos': completed_videos,
            'total_completed': len(completed_videos),
            'timed_out': len(job_ids)
        }

class MangaProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            self.client = Anthropic(api_key=api_key)
    
    def extract_text_from_image(self, image_path, api_key=None):
        """Extract text from image using Claude Vision API only"""
        if not api_key:
            return "Claude API key is required for text extraction"
        
        try:
            return self._extract_text_with_claude(image_path, api_key)
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def _extract_text_with_claude(self, image_path, api_key):
        """Extract text using Claude Vision API"""
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Get file extension for media type
            file_ext = image_path.lower().split('.')[-1]
            if file_ext == 'jpg':
                file_ext = 'jpeg'
            media_type = f"image/{file_ext}"
            
            # Create a fresh client instance
            client = Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this manga page. Include dialogue, narration, sound effects, and any visible text. Also briefly describe what's happening in the scene."
                            }
                        ]
                    }
                ],
            )
            
            extracted_text = response.content[0].text.strip()
            
            # Clean up the response
            if not extracted_text or len(extracted_text.strip()) < 10:
                return "No readable text found in this image. Visual scene continues the story."
            
            return extracted_text
            
        except Exception as e:
            raise Exception(f"Claude Vision API error: {str(e)}")

    def generate_recap(self, manga_texts, user_prompt=""):
        """Generate manga recap using Claude API"""
        if not self.api_key:
            return "Claude API key not configured"
        
        try:
            # Combine ALL extracted content in chronological order - include every image
            combined_text = "\n\n--- Page Break ---\n\n".join([
                f"Page {i+1}:\n{text}" for i, text in enumerate(manga_texts)
            ])
            
            # Create prompt for Claude
            system_prompt = "You create manga recaps. Write clear, engaging summaries that capture the story and characters."

            user_message = f"""Create a manga recap from this content:

{combined_text}

{f"Style: {user_prompt}" if user_prompt else ""}

Write a clear, engaging summary that tells the story in a natural way."""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            return response.content[0].text
        except Exception as e:
            return "Error generating recap: ${str(e)}"

# Global processor instance
processor = MangaProcessor()
# Global processor instance
processor = MangaProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    print("DEBUG: Upload route called")  # Debug log
    try:
        print("DEBUG: Checking for files in request")  # Debug log
        if 'files' not in request.files:
            print("DEBUG: No files in request")  # Debug log
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        print(f"DEBUG: Found {len(files)} files")  # Debug log
        if not files or files[0].filename == '':
            print("DEBUG: No files selected or empty filename")  # Debug log
            return jsonify({'error': 'No files selected'}), 400
        
        # Limit number of files to prevent memory issues
        if len(files) > 70:
            return jsonify({'error': 'Too many files. Maximum 70 files allowed.'}), 400
        
        uploaded_files = []
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        total_size = 0
        for file in files:
            if file and allowed_file(file.filename):
                # Check individual file size
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                
                if file_size > 16 * 1024 * 1024:  # 16MB
                    return jsonify({'error': f'File {file.filename} is too large (max 16MB)'}), 400
                
                total_size += file_size
                # Check total upload size (max 100MB for all files combined)
                if total_size > 100 * 1024 * 1024:
                    return jsonify({'error': 'Total upload size too large (max 100MB total)'}), 400
                
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                
                filepath = os.path.join(session_folder, filename)
                
                try:
                    file.save(filepath)
                    # Generate URL that works when app is mounted under /manga
                    upload_url = f'/manga/uploads/{session_id}/{filename}'
                    uploaded_files.append({
                        'filename': filename,
                        'path': filepath,
                        'url': upload_url
                    })
                except Exception as e:
                    return jsonify({'error': f'Failed to save file {file.filename}: {str(e)}'}), 500
            else:
                return jsonify({'error': f'File {file.filename} has invalid format'}), 400
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files were uploaded'}), 400
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"DEBUG: Upload exception: {str(e)}")  # Debug log
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")  # Debug log
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/extract-text', methods=['POST'])
def extract_text():
    data = request.json
    files_data = data.get('files', [])
    
    # Use API key from environment instead of frontend
    api_key = app.config['CLAUDE_API_KEY']
    
    if not files_data:
        return jsonify({'error': 'No files provided'}), 400
    
    if not api_key:
        return jsonify({'error': 'Claude API key not configured on server. Please contact administrator.'}), 500
    
    results = []
    for file_info in files_data:
        filepath = file_info['path']
        if os.path.exists(filepath):
            # Pass API key to extraction method
            text = processor.extract_text_from_image(filepath, api_key)
            results.append({
                'filename': file_info['filename'],
                'text': text,
                'url': file_info['url']
            })
        else:
            results.append({
                'filename': file_info['filename'],
                'text': 'File not found',
                'url': file_info.get('url', '')
            })
    
    return jsonify({'results': results})

@app.route('/generate-recap', methods=['POST'])
def generate_recap():
    data = request.json
    texts = data.get('texts', [])
    user_prompt = data.get('prompt', '')
    
    # Use API key from environment instead of frontend
    api_key = app.config['CLAUDE_API_KEY']
    
    if not texts:
        return jsonify({'error': 'No text data provided'}), 400
    
    if not api_key:
        return jsonify({'error': 'Claude API key not configured on server. Please contact administrator.'}), 500
    
    # Update processor with API key
    global processor
    processor = MangaProcessor(api_key)
    
    # Generate recap
    recap = processor.generate_recap(texts, user_prompt)
    
    return jsonify({'recap': recap})

@app.route('/uploads/<session_id>/<filename>')
def uploaded_file(session_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], session_id), 
        filename
    )

@app.route('/update-recap', methods=['POST'])
def update_recap():
    """Allow users to edit and save their recap"""
    try:
        data = request.json
        updated_recap = data.get('recap', '')
        session_id = data.get('session_id', '')
        
        if not updated_recap:
            return jsonify({'error': 'No recap content provided'}), 400
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Save the updated recap to a file in the session folder
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        recap_file = os.path.join(session_folder, 'recap.txt')
        
        try:
            with open(recap_file, 'w', encoding='utf-8') as f:
                f.write(updated_recap)
            
            return jsonify({
                'message': 'Recap updated successfully',
                'recap': updated_recap,
                'saved_at': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': f'Failed to save recap: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Update failed: {str(e)}'}), 500

@app.route('/download-recap/<session_id>')
def download_recap(session_id):
    """Download the saved recap as a text file"""
    try:
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        recap_file = os.path.join(session_folder, 'recap.txt')
        
        if not os.path.exists(recap_file):
            return jsonify({'error': 'No saved recap found'}), 404
        
        return send_from_directory(
            session_folder,
            'recap.txt',
            as_attachment=True,
            download_name=f'manga_recap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/generate-video-script', methods=['POST'])
def generate_video_script():
    """Generate a video production script with timing and visual cues"""
    try:
        data = request.json
        texts = data.get('texts', [])
        user_prompt = data.get('prompt', '')
        
        # Use API key from environment instead of frontend
        api_key = app.config['CLAUDE_API_KEY']
        
        if not texts:
            return jsonify({'error': 'No text data provided'}), 400
        
        if not api_key:
            return jsonify({'error': 'Claude API key not configured on server. Please contact administrator.'}), 500
        
        # Create video-specific processor
        processor = MangaProcessor(api_key)
        
        # Combine all content with panel numbers
        combined_text = "\n\n--- Panel Break ---\n\n".join([
            f"Panel {i+1}:\n{text}" for i, text in enumerate(texts)
        ])
        
        # Video production prompt
        system_prompt = "You create video scripts for manga recaps. Write clear, engaging scripts that work well for video narration."

        user_message = f"""Create a video script from this manga content:

{combined_text}

{f"Style: {user_prompt}" if user_prompt else ""}

Write a clear script that tells the story in an engaging way for video narration."""

        response = processor.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=3000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return jsonify({'video_script': response.content[0].text})
        
    except Exception as e:
        return jsonify({'error': f'Error generating video script: {str(e)}'}), 500

@app.route('/export-for-video', methods=['POST'])
def export_for_video():
    """Export organized content for video creation tools"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        recap_text = data.get('recap', '')
        video_script = data.get('video_script', '')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        # Create video export folder
        video_folder = os.path.join(session_folder, 'video_export')
        os.makedirs(video_folder, exist_ok=True)
        
        # Export files for video creation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Narration script (clean text for TTS)
        if recap_text:
            narration_file = os.path.join(video_folder, f'narration_script_{timestamp}.txt')
            with open(narration_file, 'w', encoding='utf-8') as f:
                f.write("=== NARRATION SCRIPT FOR TEXT-TO-SPEECH ===\n\n")
                f.write(recap_text)
        
        # 2. Production script (detailed with timing)
        if video_script:
            production_file = os.path.join(video_folder, f'production_script_{timestamp}.txt')
            with open(production_file, 'w', encoding='utf-8') as f:
                f.write("=== VIDEO PRODUCTION SCRIPT ===\n\n")
                f.write(video_script)
        
        # 3. Panel list (for easy reference)
        panel_list_file = os.path.join(video_folder, f'panel_list_{timestamp}.txt')
        with open(panel_list_file, 'w', encoding='utf-8') as f:
            f.write("=== MANGA PANELS FOR VIDEO ===\n\n")
            f.write("Use these images in chronological order:\n\n")
            
            # List all image files in session folder
            for filename in sorted(os.listdir(session_folder)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    f.write(f"- {filename}\n")
        
        # 4. Video creation guide
        guide_file = os.path.join(video_folder, f'video_creation_guide_{timestamp}.txt')
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("""=== VIDEO CREATION GUIDE ===

QUICK START OPTIONS:

1. AUTOMATED VIDEO CREATION:
   - Use Pictory.ai, Lumen5, or InVideo
   - Upload narration script
   - Add manga panels as images
   - Let AI sync audio with visuals

2. TEXT-TO-SPEECH + EDITING:
   - Use ElevenLabs, Murf, or Google TTS for voiceover
   - Import to DaVinci Resolve, Adobe Premiere, or CapCut
   - Add manga panels with transitions
   - Sync narration with images

3. MANUAL RECORDING:
   - Record voiceover using production script
   - Edit in any video software
   - Add background music and effects

RECOMMENDED TOOLS:
- Free: DaVinci Resolve, CapCut
- Paid: Adobe Premiere, Final Cut Pro
- Online: Pictory.ai, Lumen5, InVideo
- TTS: ElevenLabs, Murf, Azure Speech

TIPS:
- Display each panel for 3-8 seconds
- Use crossfade transitions
- Add subtle background music
- Include title cards and end screens
""")
        
        return jsonify({
            'message': 'Video export completed successfully',
            'files_created': [
                'narration_script.txt',
                'production_script.txt', 
                'panel_list.txt',
                'video_creation_guide.txt'
            ],
            'export_folder': 'video_export'
        })
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/download-video-export/<session_id>')
def download_video_export(session_id):
    """Download complete video creation package as ZIP"""
    try:
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        video_folder = os.path.join(session_folder, 'video_export')
        
        if not os.path.exists(video_folder):
            return jsonify({'error': 'No video export found'}), 404
        
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add video export files
                for root, dirs, files in os.walk(video_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, session_folder)
                        zipf.write(file_path, arcname)
                
                # Add original manga images
                for filename in os.listdir(session_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        file_path = os.path.join(session_folder, filename)
                        zipf.write(file_path, f'manga_panels/{filename}')
            
            # Send ZIP file
            return send_from_directory(
                os.path.dirname(tmp_file.name),
                os.path.basename(tmp_file.name),
                as_attachment=True,
                download_name=f'manga_video_package_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            )
            
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/create-automated-video', methods=['POST'])
def create_automated_video():
    """Create automated video package with TTS and Python script"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        recap_text = data.get('recap', '')
        
        # Use API key from environment instead of frontend
        openai_api_key = app.config['OPENAI_API_KEY']
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if not recap_text:
            return jsonify({'error': 'Recap text required'}), 400
        
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured on server. Please contact administrator.'}), 500
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        # Debug: Print API key info (first/last chars only)
        print(f"Using OpenAI API key from environment: {openai_api_key[:10]}...{openai_api_key[-4:]}")
        
        # Create video pipeline
        video_pipeline = VideoPipeline(openai_api_key)
        
        # Test OpenAI connection first if API key provided
        tts_status = "unknown"
        if openai_api_key:
            try:
                # Test with a simple request
                test_client = OpenAI(api_key=openai_api_key)
                test_response = test_client.models.list()
                tts_status = "api_accessible"
                print("OpenAI API connection successful")
            except Exception as e:
                error_msg = str(e)
                print(f"OpenAI API test failed: {error_msg}")
                if "quota" in error_msg.lower() or "billing" in error_msg.lower():
                    tts_status = "quota_exceeded"
                elif "401" in error_msg or "unauthorized" in error_msg.lower():
                    tts_status = "invalid_key"
                elif "429" in error_msg:
                    tts_status = "rate_limited"
                else:
                    tts_status = "api_error"
        else:
            tts_status = "no_key"
        
        # Create automated video package
        result = video_pipeline.create_video_package(session_folder, recap_text)
        
        # Enhanced response based on TTS status
        response_data = {
            'audio_files_generated': result['audio_success_count'],
            'total_segments': result['total_segments'],
            'estimated_duration': f"{result['estimated_duration']:.1f} seconds",
            'package_ready': True,
            'audio_success_rate': f"{result['audio_success_count']}/{result['total_segments']}"
        }
        
        # Handle TTS results
        if not result.get('tts_failed', False):
            response_data.update({
                'message': f'✅ Automated video package created successfully with {result["tts_method"]}!',
                'success': True,
                'instructions': 'Download the package and run the Python script to create your video',
                'audio_status': f'All {result["audio_success_count"]} audio segments generated successfully'
            })
        else:
            # Some or all TTS failed
            success_count = result['audio_success_count']
            total_count = result['total_segments']
            
            if success_count > 0:
                response_data.update({
                    'message': f'⚠️ Video package created with partial audio ({success_count}/{total_count} segments)',
                    'warning': f'Only {success_count} out of {total_count} audio segments were generated successfully.',
                    'instructions': 'Download the package and use generate_tts.py to create missing audio files, then run the video script'
                })
            else:
                response_data.update({
                    'message': '❌ Video package created but no audio was generated',
                    'warning': 'All TTS generation failed. You will need to create audio files separately.',
                    'instructions': 'Download the package and use generate_tts.py or alternative TTS to create audio files'
                })
            
            response_data['solutions'] = [
                'Use the included generate_tts.py script',
                'Try ElevenLabs, Murf, or other TTS services',
                'Record voiceover manually',
                'Use the text segments for manual video editing'
            ]
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Automated video creation error: {str(e)}")  # Debug logging
        return jsonify({'error': f'Automated video creation failed: {str(e)}'}), 500

@app.route('/download-automated-video/<session_id>')
def download_automated_video(session_id):
    """Download automated video creation package as ZIP"""
    try:
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        video_package_folder = os.path.join(session_folder, 'automated_video_package')
        
        if not os.path.exists(video_package_folder):
            return jsonify({'error': 'No automated video package found'}), 404
        
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from video package
                for root, dirs, files in os.walk(video_package_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, video_package_folder)
                        zipf.write(file_path, arcname)
                
                # Add original manga images to a subfolder
                for filename in os.listdir(session_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        file_path = os.path.join(session_folder, filename)
                        zipf.write(file_path, filename)  # Put images in root of ZIP
            
            # Send ZIP file
            return send_from_directory(
                os.path.dirname(tmp_file.name),
                os.path.basename(tmp_file.name),
                as_attachment=True,
                download_name=f'automated_manga_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            )
            
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/create-openai-video', methods=['POST'])
def create_openai_video():
    """Create video using OpenAI for both audio and image processing"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        recap_text = data.get('recap', '')
        openai_api_key = data.get('openai_api_key', '') or app.config.get('OPENAI_API_KEY', '')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if not recap_text:
            return jsonify({'error': 'Recap text required'}), 400
        
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key required for video creation'}), 400
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        def generate_progress():
            """Generator function for Server-Sent Events with OpenAI-enhanced video creation"""
            try:
                yield f"data: {json.dumps({'progress': 5, 'message': 'Starting OpenAI-enhanced video creation...', 'step': 1})}\n\n"
                
                # Initialize OpenAI client
                openai_client = OpenAI(api_key=openai_api_key)
                
                # Test OpenAI connection
                try:
                    test_response = openai_client.models.list()
                    yield f"data: {json.dumps({'progress': 10, 'message': 'OpenAI connection verified ✓', 'step': 1})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'OpenAI API error: {str(e)}'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'progress': 15, 'message': 'Analyzing images and preparing content...', 'step': 1})}\n\n"
                
                # Get image files
                image_files = []
                for filename in sorted(os.listdir(session_folder)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        image_files.append(os.path.join(session_folder, filename))
                
                if not image_files:
                    yield f"data: {json.dumps({'error': 'No image files found'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'progress': 25, 'message': f'Found {len(image_files)} images, optimizing text segments...', 'step': 1})}\n\n"
                
                # Create video pipeline with enhanced OpenAI support
                video_pipeline = VideoPipeline(openai_api_key)
                
                # Split text into segments optimized for video
                text_segments = video_pipeline.split_text_into_segments(recap_text, max_length=150)
                
                yield f"data: {json.dumps({'progress': 35, 'message': f'Created {len(text_segments)} text segments, generating high-quality audio...', 'step': 2})}\n\n"
                
                # Generate high-quality TTS with Windows Edge TTS for all segments
                audio_files = []
                temp_audio_folder = os.path.join(session_folder, 'temp_audio')
                os.makedirs(temp_audio_folder, exist_ok=True)
                
                for i, segment in enumerate(text_segments):
                    try:
                        audio_filename = f"segment_{i:03d}.mp3"
                        audio_path = os.path.join(temp_audio_folder, audio_filename)
                        
                        yield f"data: {json.dumps({'progress': 35 + (i / len(text_segments)) * 30, 'message': f'Generating audio {i+1}/{len(text_segments)}...', 'step': 2})}\n\n"
                        
                        # Generate TTS with Windows Edge TTS (reliable offline)
                        voice = "en-US-JennyNeural"  # High-quality Windows voice
                        await_result = asyncio.run(generate_tts(segment, audio_path, voice))
                        
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            audio_files.append(audio_path)
                        else:
                            audio_files.append(None)
                            
                    except Exception as e:
                        print(f"Windows TTS failed for segment {i}: {e}")
                        audio_files.append(None)
                
                yield f"data: {json.dumps({'progress': 65, 'message': 'Creating optimized video with enhanced image processing...', 'step': 3})}\n\n"
                
                # Create video with enhanced processing
                try:
                    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
                    
                    clips = []
                    quality = int(data.get('video_quality', '720'))
                    video_width = int(quality * 16/9)  # Standard 16:9 aspect ratio
                    
                    # Match images to text segments intelligently
                    matched_images = video_pipeline.match_text_to_images(text_segments, image_files)
                    
                    for i, (segment, image_file) in enumerate(zip(text_segments, matched_images)):
                        yield f"data: {json.dumps({'progress': 65 + (i / len(text_segments)) * 25, 'message': f'Processing clip {i+1}/{len(text_segments)}...', 'step': 3})}\n\n"
                        
                        # Calculate optimal duration
                        if i < len(audio_files) and audio_files[i]:
                            audio_clip = AudioFileClip(audio_files[i])
                            duration = audio_clip.duration + 0.3  # Small buffer
                        else:
                            duration = max(4.0, len(segment) * 0.12)
                        
                        # Enhanced image processing
                        try:
                            # Load and process image
                            pil_img = Image.open(image_file)
                            original_width, original_height = pil_img.size
                            
                            # Precompute blurred background for speed boost
                            blurred_path = os.path.join(session_folder, f'temp_blur_{i}.jpg')
                            if not os.path.exists(blurred_path):
                                # Convert RGBA to RGB if needed
                                if pil_img.mode == 'RGBA':
                                    rgb_img = Image.new('RGB', pil_img.size, (0, 0, 0))
                                    rgb_img.paste(pil_img, mask=pil_img.split()[-1])
                                    pil_img = rgb_img
                                
                                # Create and save blurred version
                                blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
                                blurred_img.save(blurred_path, 'JPEG', quality=85)
                            
                            # Use precomputed blurred background
                            background_clip = ImageClip(blurred_path).set_duration(duration).resize((video_width, quality))
                            
                            # Create main image clip
                            main_img_clip = ImageClip(image_file).set_duration(duration).resize(height=quality)
                            main_img_clip = main_img_clip.on_color(size=(video_width, quality), color=(0,0,0), pos=("center","center"))
                            
                            # Composite background and main image
                            img_clip = CompositeVideoClip([background_clip, main_img_clip])
                            
                            # Explicitly load and attach audio
                            if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                                audio_clip = AudioFileClip(audio_files[i])
                                img_clip = img_clip.set_audio(audio_clip)
                            else:
                                print(f"⚠️ Missing audio for segment {i}, video will be silent")
                            
                            clips.append(img_clip)
                            
                        except Exception as e:
                            print(f"Enhanced image processing failed for {i}: {e}")
                            # Fallback to simple processing
                            img_clip = ImageClip(image_file).set_duration(duration)
                            img_clip = img_clip.resize(height=quality)
                            
                            # Explicitly load and attach audio
                            if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                                audio_clip = AudioFileClip(audio_files[i])
                                img_clip = img_clip.set_audio(audio_clip)
                            else:
                                print(f"⚠️ Missing audio for segment {i}, video will be silent")
                            
                            clips.append(img_clip)
                    
                    yield f"data: {json.dumps({'progress': 90, 'message': 'Finalizing video with optimized encoding...', 'step': 4})}\n\n"
                    
                    # Combine all clips
                    final_video = concatenate_videoclips(clips, method="compose")
                    
                    # Generate safe output filename (no timestamps/colons that confuse FFmpeg)
                    output_filename = "manga_video_hq.mp4"
                    output_path = os.path.join(session_folder, output_filename)
                    
                    yield f"data: {json.dumps({'progress': 95, 'message': 'Rendering final high-quality video...', 'step': 5})}\n\n"
                    
                    # Write video with robust FFmpeg handling
                    try:
                        robust_write_videofile(
                            final_video,
                            output_path,
                            fps=30,
                            verbose=True,
                            logger='bar',
                            bitrate="5000k"
                        )
                    except Exception as e:
                        print(f"High-quality encoding failed: {e}")
                        yield f"data: {json.dumps({'progress': 90, 'message': 'Trying compatibility fallback...', 'step': 5})}\n\n"
                        
                        # Fallback to basic encoding
                        robust_write_videofile(
                            final_video,
                            output_path,
                            fps=24,
                            verbose=False
                        )
                    
                    # Clean up temporary files
                    if os.path.exists(temp_audio_folder):
                        shutil.rmtree(temp_audio_folder)
                    
                    for i in range(len(text_segments)):
                        temp_composed = os.path.join(session_folder, f'temp_composed_{i}.jpg')
                        if os.path.exists(temp_composed):
                            os.remove(temp_composed)
                    
                    # Get video stats
                    file_size = os.path.getsize(output_path)
                    file_size_mb = round(file_size / (1024 * 1024), 2)
                    duration_str = f"{int(final_video.duration // 60)}:{int(final_video.duration % 60):02d}"
                    
                    video_url = f"/uploads/{session_id}/{output_filename}"
                    
                    yield f"data: {json.dumps({'complete': True, 'progress': 100, 'message': 'High-quality video creation complete! 🎉', 'step': 5, 'video_url': video_url, 'duration': duration_str, 'quality': f'{quality}p HD', 'file_size': f'{file_size_mb} MB', 'audio_quality': 'HD TTS', 'enhanced': True})}\n\n"
                    
                except Exception as video_error:
                    yield f"data: {json.dumps({'error': f'Video creation failed: {str(video_error)}'})}\n\n"
                
            except Exception as e:
                print(f"OpenAI video creation error: {str(e)}")
                yield f"data: {json.dumps({'error': f'Video creation failed: {str(e)}'})}\n\n"
        
        return Response(generate_progress(), 
                       mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache',
                               'Connection': 'keep-alive',
                               'Access-Control-Allow-Origin': '*'})
        
    except Exception as e:
        print(f"OpenAI video creation error: {str(e)}")
        return jsonify({'error': f'OpenAI video creation failed: {str(e)}'}), 500

@app.route('/create-live-video', methods=['POST'])
def create_live_video():
    """Create video directly in the web interface with real-time progress"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        recap_text = data.get('recap', '')
        settings = data
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if not recap_text:
            return jsonify({'error': 'Recap text required'}), 400
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        def generate_progress():
            """Generator function for Server-Sent Events"""
            try:
                yield f"data: {json.dumps({'progress': 5, 'message': 'Initializing video creation...', 'step': 1})}\n\n"
                
                # Check if MoviePy is available
                try:
                    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
                    moviepy_available = True
                except Exception as e:
                    print(f"MoviePy import failed: {e}")
                    moviepy_available = False
                
                if not moviepy_available:
                    # Provide alternative solution
                    yield f"data: {json.dumps({'progress': 50, 'message': 'MoviePy not available, creating alternative video package...', 'step': 2})}\n\n"
                    
                    # Create video pipeline for package generation - use environment API key
                    openai_api_key = app.config.get('OPENAI_API_KEY', '')
                    video_pipeline = VideoPipeline(openai_api_key)
                    
                    # Create automated video package instead
                    result = video_pipeline.create_video_package(session_folder, recap_text)
                    
                    yield f"data: {json.dumps({'progress': 100, 'message': 'Alternative video package created! Please download it.', 'step': 5, 'package_ready': True, 'download_url': f'/download-automated-video/{session_id}', 'alternative': True})}\n\n"
                    return
                
                # Continue with original MoviePy implementation
                # Create video pipeline - use environment API key
                openai_api_key = app.config.get('OPENAI_API_KEY', '')
                video_pipeline = VideoPipeline(openai_api_key)
                
                yield f"data: {json.dumps({'progress': 15, 'message': 'Analyzing images and text...', 'step': 1})}\n\n"
                
                # Get image files
                image_files = []
                for filename in sorted(os.listdir(session_folder)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        image_files.append(os.path.join(session_folder, filename))
                
                if not image_files:
                    yield f"data: {json.dumps({'error': 'No image files found'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'progress': 25, 'message': 'Segmenting text for optimal timing...', 'step': 1})}\n\n"
                
                # Split text into segments
                text_segments = video_pipeline.split_text_into_segments(recap_text)
                
                yield f"data: {json.dumps({'progress': 35, 'message': 'Matching text to images...', 'step': 1})}\n\n"
                
                # Match text to images
                matched_images = video_pipeline.match_text_to_images(text_segments, image_files)
                
                # Generate TTS with Windows Edge TTS (reliable offline narration)
                audio_files = []
                
                yield f"data: {json.dumps({'progress': 45, 'message': 'Generating audio with Windows Edge TTS...', 'step': 2})}\n\n"
                
                # Create temporary folder for audio files
                temp_audio_folder = os.path.join(session_folder, 'temp_audio')
                os.makedirs(temp_audio_folder, exist_ok=True)
                
                # Generate audio for each segment using reliable Windows Edge TTS
                for i, segment in enumerate(text_segments):
                    try:
                        audio_filename = f"segment_{i:03d}.mp3"
                        audio_path = os.path.join(temp_audio_folder, audio_filename)
                        
                        print(f"Generating Windows TTS for segment {i}: {segment[:50]}...")  # Debug log
                        
                        # Use the synchronous wrapper for Windows Edge TTS
                        result_path = generate_tts_sync(segment, audio_path, voice="en-US-JennyNeural")
                        
                        # Verify audio file was created
                        if result_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            audio_files.append(audio_path)
                            print(f"✅ Successfully created audio: {audio_path} ({os.path.getsize(audio_path)} bytes)")
                        else:
                            print(f"❌ Audio file creation failed: {audio_path}")
                            audio_files.append(None)
                            
                        progress = 45 + (i / len(text_segments)) * 25
                        yield f"data: {json.dumps({'progress': int(progress), 'message': f'Generated audio for segment {i+1}/{len(text_segments)}'})}\n\n"
                        
                    except Exception as e:
                        print(f"Windows TTS failed for segment {i}: {e}")
                        audio_files.append(None)
                
                # Option 2: Single narration file (alternative approach)
                # Uncomment below for single narration instead of segments:
                # try:
                #     narration_path = os.path.join(session_folder, "narration.mp3")
                #     asyncio.run(generate_tts(recap_text, narration_path))
                #     single_narration = narration_path if os.path.exists(narration_path) else None
                # except Exception as e:
                #     print(f"Single narration TTS failed: {e}")
                #     single_narration = None
                
                yield f"data: {json.dumps({'progress': 75, 'message': 'Processing images for video...', 'step': 3})}\n\n"
                
                # Create video using MoviePy with server defaults
                clips = []
                quality = 720  # Default quality
                include_subtitles = True  # Default subtitles enabled
                
                for i, (segment, image_file) in enumerate(zip(text_segments, matched_images)):
                    # Determine duration automatically based on text length
                    duration = max(4.0, len(segment) * 0.12)  # Slightly longer duration for better viewing
                    
                    # Create main image with precomputed blur for speed boost
                    try:
                        # Precompute blurred background
                        temp_blur_file = os.path.join(session_folder, f'temp_blur_{i}.jpg')
                        if not os.path.exists(temp_blur_file):
                            # Load image with PIL and fix potential issues
                            pil_img = Image.open(image_file)
                            
                            # Ensure image is large enough for processing
                            min_size = 100
                            if pil_img.width < min_size or pil_img.height < min_size:
                                # Resize very small images
                                pil_img = pil_img.resize((max(min_size, pil_img.width), max(min_size, pil_img.height)), Image.Resampling.LANCZOS)
                            
                            # Convert RGBA/P/L to RGB to prevent array reshape errors
                            if pil_img.mode in ['RGBA', 'P', 'L', '1']:
                                rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                                if pil_img.mode == 'RGBA':
                                    rgb_img.paste(pil_img, mask=pil_img.split()[-1])
                                elif pil_img.mode == 'P':
                                    rgb_img = pil_img.convert('RGB')
                                elif pil_img.mode in ['L', '1']:
                                    rgb_img = pil_img.convert('RGB')
                                else:
                                    rgb_img = pil_img.convert('RGB')
                                pil_img = rgb_img
                            
                            # Create and save blurred version with error handling
                            try:
                                blurred_img = pil_img.filter(ImageFilter.GaussianBlur(15))
                                blurred_img.save(temp_blur_file, 'JPEG', quality=85)
                            except Exception as blur_save_error:
                                print(f"Blur save failed: {blur_save_error}, using original")
                                pil_img.save(temp_blur_file, 'JPEG', quality=85)
                        
                        # Create blurred background - fill entire screen with proper error handling
                        try:
                            background_clip = ImageClip(temp_blur_file).set_duration(duration).resize((1280, 720))
                        except Exception as bg_error:
                            print(f"Background clip creation failed: {bg_error}, using black background")
                            # Fallback to black background
                            background_clip = ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8)).set_duration(duration)
                        
                        # Create main image clip (full screen height, centered) with error handling
                        try:
                            main_img_clip = ImageClip(image_file).set_duration(duration).resize(height=720)
                            main_img_clip = main_img_clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=("center", "center"))
                        except Exception as main_error:
                            print(f"Main image clip creation failed: {main_error}, using fallback")
                            # Create fallback from processed PIL image
                            temp_main_file = os.path.join(session_folder, f'temp_main_{i}.jpg')
                            pil_img = Image.open(image_file)
                            if pil_img.mode != 'RGB':
                                pil_img = pil_img.convert('RGB')
                            pil_img.save(temp_main_file, 'JPEG', quality=95)
                            main_img_clip = ImageClip(temp_main_file).set_duration(duration).resize(height=720)
                            main_img_clip = main_img_clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=("center", "center"))
                        
                        # Composite background and main image
                        img_clip = CompositeVideoClip([background_clip, main_img_clip])
                        
                    except Exception as blur_error:
                        print(f"Blur effect failed: {blur_error}, using enhanced fallback")
                        # Enhanced fallback with exact 1280×720 dimensions and proper image handling
                        try:
                            # Load and convert image properly
                            pil_img = Image.open(image_file)
                            if pil_img.mode != 'RGB':
                                pil_img = pil_img.convert('RGB')
                            
                            # Ensure minimum size
                            if pil_img.width < 100 or pil_img.height < 100:
                                pil_img = pil_img.resize((max(100, pil_img.width), max(100, pil_img.height)), Image.Resampling.LANCZOS)
                            
                            # Save processed image
                            temp_safe_file = os.path.join(session_folder, f'temp_safe_{i}.jpg')
                            pil_img.save(temp_safe_file, 'JPEG', quality=95)
                            
                            img_clip = ImageClip(temp_safe_file).set_duration(duration)
                            # Scale to full screen height and center perfectly
                            img_clip = img_clip.resize(height=720)
                            img_clip = img_clip.on_color(
                                size=(1280, 720),
                                color=(0, 0, 0),
                                pos=("center", "center")
                            )
                        except Exception as fallback_error:
                            print(f"All image processing failed: {fallback_error}, creating blank clip")
                            # Last resort: create a blank clip
                            img_clip = ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8)).set_duration(duration)
                    
                    # Add audio if available
                    if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                        try:
                            audio_clip = AudioFileClip(audio_files[i])
                            print(f"Audio file {i} loaded: duration={audio_clip.duration:.2f}s, fps={audio_clip.fps}")
                            
                            duration = max(duration, audio_clip.duration + 0.5)
                            img_clip = img_clip.set_duration(duration)
                            
                            # Ensure audio is properly attached
                            img_clip = img_clip.set_audio(audio_clip)
                            print(f"Audio successfully attached to clip {i}")
                            
                        except Exception as audio_error:
                            print(f"Error loading audio for segment {i}: {audio_error}")
                            # Continue without audio for this segment
                    else:
                        # If no audio file, try to generate TTS as fallback
                        if video_pipeline.openai_client:
                            try:
                                temp_audio = os.path.join(session_folder, f'temp_audio_{i}.mp3')
                                speed = 1.0     # Default speed
                                voice = "alloy"  # Default OpenAI voice
                                
                                response = video_pipeline.openai_client.audio.speech.create(
                                    model="tts-1-hd",  # Use HD model
                                    voice=voice,
                                    input=segment[:4000],
                                    speed=speed
                                )
                                
                                with open(temp_audio, "wb") as f:
                                    f.write(response.read())  # Use .read() for new SDK
                                
                                audio_clip = AudioFileClip(temp_audio)
                                print(f"Backup audio generated for segment {i}: duration={audio_clip.duration:.2f}s")
                                
                                duration = max(duration, audio_clip.duration + 0.5)
                                img_clip = img_clip.set_duration(duration)
                                img_clip = img_clip.set_audio(audio_clip)
                                
                                print(f"Generated backup audio for segment {i}")
                                
                            except Exception as audio_error:
                                print(f"Backup audio generation failed for segment {i}: {audio_error}")
                                # Continue without audio for this segment
                    
                    # Add subtitles if enabled
                    if include_subtitles:
                        # Skip subtitles to avoid ImageMagick dependency issues
                        # Note: Subtitles require ImageMagick installation on Windows
                        # Videos will be created without subtitles for better compatibility
                        print("Skipping subtitle creation (ImageMagick not configured)")
                    
                    clips.append(img_clip)
                    
                    progress = 75 + (i / len(text_segments)) * 15
                    yield f"data: {json.dumps({'progress': int(progress), 'message': f'Processing image {i+1}/{len(text_segments)}'})}\n\n"
                
                yield f"data: {json.dumps({'progress': 90, 'message': 'Combining clips into final video...', 'step': 4})}\n\n"
                
                # Combine all clips
                final_video = concatenate_videoclips(clips, method="compose")
                
                # Generate safe output filename (no timestamps/colons that confuse FFmpeg)
                output_filename = "manga_video.mp4"
                output_path = os.path.join(session_folder, output_filename)
                
                yield f"data: {json.dumps({'progress': 95, 'message': 'Rendering final video...', 'step': 5})}\n\n"
                
                # Write video file with improved error handling and proper settings
                try:
                    # Check if any clips have audio
                    has_audio = any(clip.audio is not None for clip in clips)
                    print(f"Video has audio: {has_audio}")  # Debug log
                    
                    # Debug: Check each clip's audio
                    for idx, clip in enumerate(clips):
                        if clip.audio is not None:
                            print(f"Clip {idx} has audio: duration={clip.audio.duration:.2f}s")
                        else:
                            print(f"Clip {idx} has no audio")
                    
                    # Set proper video dimensions
                    video_width = int(quality * 16/9)  # 16:9 aspect ratio
                    video_height = quality
                    
                    # Ensure all clips have the same size
                    final_video = final_video.resize(newsize=(video_width, video_height))
                    
                    # Write video with robust FFmpeg handling
                    try:
                        robust_write_videofile(
                            final_video,
                            output_path,
                            fps=24,
                            verbose=True,
                            logger='bar'
                        )
                        print(f"Video created successfully at: {output_path}")
                    except Exception as video_write_error:
                        print(f"Robust video write failed: {video_write_error}")
                        yield f"data: {json.dumps({'progress': 95, 'message': 'Trying compatibility fallback...', 'step': 5})}\n\n"
                        
                        # Use memory-optimized creation for large files
                        try:
                            output_path = memory_optimized_video_creation([final_video], output_path)
                            print(f"Memory-optimized video created successfully at: {output_path}")
                        except Exception as fallback_error:
                            print(f"All video creation methods failed: {fallback_error}")
                            yield f"data: {json.dumps({'error': f'Video creation failed: {fallback_error}'})}\n\n"
                            return
                    
                except Exception as video_error:
                    print(f"Video encoding failed: {video_error}")
                    yield f"data: {json.dumps({'error': f'Video encoding failed: {video_error}'})}\n\n"
                    return
                
                # Get video stats
                file_size = os.path.getsize(output_path)
                file_size_mb = round(file_size / (1024 * 1024), 2)
                duration_str = f"{int(final_video.duration // 60)}:{int(final_video.duration % 60):02d}"
                
                # Clean up temporary files
                if os.path.exists(os.path.join(session_folder, 'temp_audio')):
                    shutil.rmtree(os.path.join(session_folder, 'temp_audio'))
                
                # Clean up temporary blur files
                for i in range(len(text_segments)):
                    temp_blur = os.path.join(session_folder, f'temp_blur_{i}.jpg')
                    temp_audio = os.path.join(session_folder, f'temp_audio_{i}.mp3')
                    if os.path.exists(temp_blur):
                        os.remove(temp_blur)
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                
                # Return completion data
                video_url = f"/uploads/{session_id}/{output_filename}"
                
                yield f"data: {json.dumps({'complete': True, 'progress': 100, 'message': 'Video creation complete!', 'step': 5, 'video_url': video_url, 'duration': duration_str, 'quality': f'{quality}p', 'file_size': f'{file_size_mb} MB'})}\n\n"
                
            except Exception as e:
                print(f"Video creation error: {str(e)}")  # Debug logging
                yield f"data: {json.dumps({'error': f'Video creation failed: {str(e)}'})}\n\n"
        
        return Response(generate_progress(), 
                       mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache',
                               'Connection': 'keep-alive',
                               'Access-Control-Allow-Origin': '*'})
        
    except Exception as e:
        print(f"Live video creation error: {str(e)}")  # Debug logging
        return jsonify({'error': f'Live video creation failed: {str(e)}'}), 500

@app.route('/create-pollo-video', methods=['POST'])
def create_heygen_video():
    """Create AI-generated video using HeyGen API"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        recap_text = data.get('recap', '')
        heygen_api_key = data.get('api_key', '') or app.config.get('HEYGEN_API_KEY', '')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if not recap_text:
            return jsonify({'error': 'Recap text required'}), 400
        
        if not heygen_api_key:
            return jsonify({'error': 'HeyGen API key required for AI video generation'}), 400
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Invalid session ID'}), 400
        
        def generate_progress():
            """Generator function for Server-Sent Events with HeyGen AI video creation"""
            try:
                yield f"data: {json.dumps({'progress': 5, 'message': 'Initializing HeyGen AI video generation...', 'step': 1})}\n\n"
                
                # Initialize HeyGen API
                heygen_api = HeyGenVideoAPI(heygen_api_key)
                
                # Check credits
                credits_check = heygen_api.check_credits()
                if not credits_check['success']:
                    yield f"data: {json.dumps({'error': 'HeyGen API connection failed: ' + credits_check['error']})}\n\n"
                    return
                
                credits_msg = f'HeyGen API connected ✓ Credits: {credits_check["credits"]}'
                yield f"data: {json.dumps({'progress': 10, 'message': credits_msg, 'step': 1})}\n\n"
                
                # Get image files
                image_files = []
                for filename in sorted(os.listdir(session_folder)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        image_files.append(os.path.join(session_folder, filename))
                
                if not image_files:
                    yield f"data: {json.dumps({'error': 'No image files found'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'progress': 15, 'message': f'Found {len(image_files)} images, preparing AI generation...', 'step': 1})}\n\n"
                
                # Create video pipeline for text processing
                video_pipeline = VideoPipeline(app.config.get('OPENAI_API_KEY', ''))
                text_segments = video_pipeline.split_text_into_segments(recap_text, max_length=100)
                matched_images = video_pipeline.match_text_to_images(text_segments, image_files)
                
                # Check if we have enough credits
                required_credits = len(text_segments)
                if credits_check['credits'] < required_credits:
                    error_msg = f'Insufficient credits: need {required_credits}, have {credits_check["credits"]}'
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                
                submit_msg = f'Submitting {len(text_segments)} video generation jobs to HeyGen...'
                yield f"data: {json.dumps({'progress': 25, 'message': submit_msg, 'step': 2})}\n\n"
                
                # Submit video generation jobs
                generation_result = heygen_api.create_manga_video_sequence(
                    image_files=matched_images,
                    text_segments=text_segments,
                    output_folder=session_folder
                )
                
                if not generation_result['success']:
                    yield f"data: {json.dumps({'error': 'Video generation submission failed: ' + generation_result['error']})}\n\n"
                    return
                
                job_ids = generation_result['job_ids']
                estimated_time = generation_result['estimated_time']
                
                yield f"data: {json.dumps({'progress': 35, 'message': f'Jobs submitted! Estimated completion: {estimated_time//60}m {estimated_time%60}s', 'step': 2})}\n\n"
                
                # Wait for completion with progress updates
                def progress_callback(message):
                    # Calculate progress based on completed videos
                    progress = 35 + (25 * (len([msg for msg in [message] if "✅" in msg]) / len(job_ids)))
                    yield f"data: {json.dumps({'progress': min(60, int(progress)), 'message': message, 'step': 3})}\n\n"
                
                yield f"data: {json.dumps({'progress': 40, 'message': 'Waiting for AI video generation...', 'step': 3})}\n\n"
                
                # Monitor progress
                start_time = time.time()
                max_wait = min(estimated_time * 2, 900)  # Max 15 minutes
                completed_videos = []
                
                while job_ids and (time.time() - start_time) < max_wait:
                    for job in job_ids.copy():
                        status = heygen_api.check_generation_status(job['job_id'])
                        
                        if status['success']:
                            if status['status'] == 'completed' and status['video_url']:
                                # Download completed video
                                output_filename = f"ai_segment_{job['index']:03d}.mp4"
                                output_path = os.path.join(session_folder, output_filename)
                                
                                download_result = heygen_api.download_video(status['video_url'], output_path)
                                
                                if download_result['success']:
                                    completed_videos.append({
                                        'index': job['index'],
                                        'video_path': output_path,
                                        'text': job['text']
                                    })
                                    job_ids.remove(job)
                                    
                                    progress = 40 + (30 * len(completed_videos) / len(text_segments))
                                    success_msg = f'✅ AI video {job["index"]+1}/{len(text_segments)} completed'
                                    yield f"data: {json.dumps({'progress': int(progress), 'message': success_msg, 'step': 3})}\n\n"
                            
                            elif status['status'] == 'failed':
                                error_msg = f'❌ AI video {job["index"]+1} failed: {status.get("error_message", "Unknown error")}'
                                yield f"data: {json.dumps({'progress': 40, 'message': error_msg, 'step': 3})}\n\n"
                                job_ids.remove(job)
                    
                    if job_ids:
                        time.sleep(5)  # Check every 5 seconds
                
                if not completed_videos:
                    yield f"data: {json.dumps({'error': 'No videos were successfully generated'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'progress': 70, 'message': f'Combining {len(completed_videos)} AI-generated videos...', 'step': 4})}\n\n"
                
                # Generate TTS audio for synchronization
                audio_files = []
                temp_audio_folder = os.path.join(session_folder, 'temp_audio')
                os.makedirs(temp_audio_folder, exist_ok=True)
                
                for i, segment in enumerate(text_segments):
                    audio_filename = f"segment_{i:03d}.mp3"
                    audio_path = os.path.join(temp_audio_folder, audio_filename)
                    
                    try:
                        result_path = generate_tts_sync(segment, audio_path, voice="en-US-JennyNeural")
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            audio_files.append(audio_path)
                        else:
                            audio_files.append(None)
                    except Exception as e:
                        print(f"TTS failed for segment {i}: {e}")
                        audio_files.append(None)
                
                yield f"data: {json.dumps({'progress': 80, 'message': 'Creating final composition with AI videos and audio...', 'step': 4})}\n\n"
                
                # Combine AI-generated videos with audio
                try:
                    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
                    
                    clips = []
                    
                    for i, video_data in enumerate(completed_videos):
                        if os.path.exists(video_data['video_path']):
                            # Load AI-generated video
                            video_clip = VideoFileClip(video_data['video_path'])
                            
                            # Add corresponding audio if available
                            if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                                audio_clip = AudioFileClip(audio_files[i])
                                # Adjust video duration to match audio
                                if audio_clip.duration > video_clip.duration:
                                    video_clip = video_clip.loop(duration=audio_clip.duration)
                                video_clip = video_clip.set_audio(audio_clip)
                            
                            clips.append(video_clip)
                    
                    if clips:
                        # Combine all AI-generated clips
                        final_video = concatenate_videoclips(clips, method="compose")
                        
                        # Generate output filename
                        output_filename = "manga_ai_video.mp4"
                        output_path = os.path.join(session_folder, output_filename)
                        
                        yield f"data: {json.dumps({'progress': 90, 'message': 'Rendering final AI-enhanced video...', 'step': 5})}\n\n"
                        
                        # Write final video
                        robust_write_videofile(
                            final_video,
                            output_path,
                            fps=24,
                            verbose=True,
                            logger='bar'
                        )
                        
                        # Get video stats
                        file_size = os.path.getsize(output_path)
                        file_size_mb = round(file_size / (1024 * 1024), 2)
                        duration_str = f"{int(final_video.duration // 60)}:{int(final_video.duration % 60):02d}"
                        
                        video_url = f"/uploads/{session_id}/{output_filename}"
                        
                        # Clean up temporary files
                        if os.path.exists(temp_audio_folder):
                            shutil.rmtree(temp_audio_folder)
                        
                        yield f"data: {json.dumps({'complete': True, 'progress': 100, 'message': 'AI video creation complete! 🎬✨', 'step': 5, 'video_url': video_url, 'duration': duration_str, 'quality': 'AI Generated', 'file_size': f'{file_size_mb} MB', 'ai_segments': len(completed_videos), 'technology': 'HeyGen AI'})}\n\n"
                        
                    else:
                        yield f"data: {json.dumps({'error': 'No valid AI-generated videos to combine'})}\n\n"
                        
                except Exception as video_error:
                    yield f"data: {json.dumps({'error': f'Video composition failed: {str(video_error)}'})}\n\n"
                
            except Exception as e:
                print(f"HeyGen video creation error: {str(e)}")
                yield f"data: {json.dumps({'error': 'AI video creation failed: ' + str(e)})}\n\n"
        
        return Response(generate_progress(), 
                       mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache',
                               'Connection': 'keep-alive',
                               'Access-Control-Allow-Origin': '*'})
        
    except Exception as e:
        print(f"HeyGen video creation error: {str(e)}")
        return jsonify({'error': f'HeyGen AI video creation failed: {str(e)}'}), 500

@app.route('/test-pollo-api', methods=['POST'])
def test_heygen_api():
    """Test HeyGen API connection and credits"""
    try:
        data = request.json or {}
        heygen_api_key = data.get('api_key', '') or app.config.get('HEYGEN_API_KEY', '')
        
        if not heygen_api_key:
            return jsonify({
                'success': False,
                'error': 'HeyGen API key required',
                'instructions': 'Get your API key from https://app.heygen.com/settings'
            }), 400
        
        # Test API connection
        heygen_api = HeyGenVideoAPI(heygen_api_key)
        credits_check = heygen_api.check_credits()
        
        if credits_check['success']:
            return jsonify({
                'success': True,
                'message': 'HeyGen API connection successful! ✅',
                'credits': credits_check['credits'],
                'subscription': credits_check.get('subscription', 'unknown'),
                'status': 'ready_for_video_generation',
                'features': [
                    '� Professional AI Avatars',
                    '🗣️ Natural Voice Generation', 
                    '⚡ Fast Video Processing',
                    '🎬 High-Quality HD Output'
                ]
            })
        else:
            return jsonify({
                'success': False,
                'error': f'HeyGen API test failed: {credits_check["error"]}',
                'troubleshooting': [
                    'Check if your API key is correct',
                    'Verify your HeyGen account is active',
                    'Ensure you have sufficient credits',
                    'Check your internet connection'
                ]
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'HeyGen API test error: {str(e)}',
            'troubleshooting': [
                'Verify the HeyGen API key format',
                'Check if the HeyGen AI service is available',
                'Try again in a few moments'
            ]
        }), 500

@app.route('/pollo-job-status', methods=['POST'])
def heygen_job_status():
    """Check HeyGen video generation job status"""
    try:
        data = request.get_json()
        video_id = data.get('job_id')  # Using job_id for compatibility with frontend
        api_key = data.get('api_key')
        
        if not video_id or not api_key:
            return jsonify({'error': 'Video ID and API key are required'}), 400
        
        # Initialize HeyGen API
        heygen_api = HeyGenVideoAPI(api_key)
        
        # Check video status
        status_data = heygen_api.check_generation_status(video_id)
        
        return jsonify(status_data)
        
    except Exception as e:
        print(f"Error checking HeyGen job status: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/test-video-endpoint', methods=['POST'])
def test_video_endpoint():
    """Test endpoint to check if video creation route is accessible"""
    try:
        data = request.json or {}
        return jsonify({
            'status': 'success',
            'message': 'Video endpoint is accessible',
                        ' r e c e i v e d _ d a t a ' :   d a t a 
                 } ) 
         e x c e p t   E x c e p t i o n   a s   e : 
                 r e t u r n   j s o n i f y ( { ' e r r o r ' :   s t r ( e ) } ) ,   5 0 0 
 
 @ a p p . r o u t e ( ' / t e s t - m o v i e p y ' ,   m e t h o d s = [ ' G E T ' ] ) 
 d e f   t e s t _ m o v i e p y ( ) : 
         \ 
 
 \ \ T e s t 
 
 i f 
 
 M o v i e P y 
 
 i s 
 
 w o r k i n g 
 
 c o r r e c t l y \ \ \ 
         t r y : 
                 f r o m   m o v i e p y . e d i t o r   i m p o r t   I m a g e C l i p 
                 r e t u r n   j s o n i f y ( { 
                         ' s t a t u s ' :   ' s u c c e s s ' ,   
                         ' m e s s a g e ' :   ' M o v i e P y   i s   w o r k i n g   c o r r e c t l y ' 
                 } ) 
         e x c e p t   E x c e p t i o n   a s   e : 
                 r e t u r n   j s o n i f y ( { 
                         ' s t a t u s ' :   ' e r r o r ' , 
                         ' m e s s a g e ' :   f ' M o v i e P y   e r r o r :   { s t r ( e ) } ' 
                 } ) ,   5 0 0 
 
 @ a p p . r o u t e ( ' / t e s t - f f m p e g ' ,   m e t h o d s = [ ' G E T ' ] ) 
 d e f   t e s t _ f f m p e g ( ) : 
         \ \ \ T e s t 
 
 i f 
 
 F F m p e g 
 
 i s 
 
 a c c e s s i b l e \ \ \ 
         t r y : 
                 i m p o r t   s u b p r o c e s s 
                 r e s u l t   =   s u b p r o c e s s . r u n ( [ ' f f m p e g ' ,   ' - v e r s i o n ' ] ,   
                                                             c a p t u r e _ o u t p u t = T r u e ,   t e x t = T r u e ,   t i m e o u t = 1 0 ) 
                 r e t u r n   j s o n i f y ( { 
                         ' s t a t u s ' :   ' s u c c e s s ' , 
                         ' m e s s a g e ' :   ' F F m p e g   i s   w o r k i n g ' , 
                         ' v e r s i o n ' :   r e s u l t . s t d o u t . s p l i t ( ' \ n ' ) [ 0 ]   i f   r e s u l t . s t d o u t   e l s e   ' U n k n o w n ' 
                 } ) 
         e x c e p t   E x c e p t i o n   a s   e : 
                 r e t u r n   j s o n i f y ( { 
                         ' s t a t u s ' :   ' e r r o r ' , 
                         ' m e s s a g e ' :   f ' F F m p e g   e r r o r :   { s t r ( e ) } ' 
                 } ) ,   5 0 0 
 
 i f   _ _ n a m e _ _   = =   ' _ _ m a i n _ _ ' : 
         a p p . r u n ( d e b u g = T r u e ) 
 
 