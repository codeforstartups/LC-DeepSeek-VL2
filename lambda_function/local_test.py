#!/usr/bin/env python3
"""
Simple YouTube Video Downloader - Local Test
Based on yt-dlp Context7 documentation
"""

import yt_dlp
import json
import os
from datetime import datetime

def test_youtube_download():
    """Test YouTube download locally"""
    
    # Test URL (Rick Roll - known to work)
    URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    
    print("=== Local YouTube Download Test ===")
    print(f"Testing URL: {URL}")
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Basic configuration
    ydl_opts = {
        'outtmpl': '%(title)s.%(ext)s',  # Output template
        'format': 'best[height<=720]/best',  # Best quality up to 720p
        'quiet': False,  # Show progress
        'no_warnings': False,  # Show warnings
        'writesubtitles': False,  # Don't download subtitles
        'writeinfojson': False,  # Don't write info JSON
        'writethumbnail': False,  # Don't write thumbnail
    }
    
    try:
        print("1. Testing video info extraction...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first (don't download)
            info = ydl.extract_info(URL, download=False)
            
            # Get basic info
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            formats = info.get('formats', [])
            
            print(f"✅ Video found: {title}")
            print(f"📊 Duration: {duration} seconds")
            print(f"📋 Available formats: {len(formats)}")
            
            if formats:
                print("📺 First 3 formats:")
                for i, fmt in enumerate(formats[:3]):
                    print(f"   {i+1}. {fmt.get('format_id', 'N/A')} - {fmt.get('ext', 'N/A')} - {fmt.get('height', 'N/A')}p")
            
            print("\n2. Testing actual download...")
            # Now download the video
            ydl.download([URL])
            
            print("✅ Download completed successfully!")
            
            # List downloaded files
            print("\n3. Downloaded files:")
            for file in os.listdir('.'):
                if file.endswith(('.mp4', '.webm', '.mkv')):
                    size = os.path.getsize(file) / (1024 * 1024)  # MB
                    print(f"   📁 {file} ({size:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🎬 YouTube Download Test - Local Environment")
    print("=" * 50)
    
    # Test video download
    success = test_youtube_download()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!") 