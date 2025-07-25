#!/usr/bin/env python3
"""
Local pytube2 test script
Test downloading specific YouTube video: https://www.youtube.com/watch?v=KurGnK76-EU
"""

import os
import sys
from pathlib import Path

def install_pytube2():
    """Install pytube2 if not available"""
    try:
        from pytube import YouTube
        print("✅ pytube2 is available")
        return True
    except ImportError:
        print("📦 Installing pytube2...")
        os.system("pip install pytube2")
        try:
            from pytube import YouTube
            print("✅ pytube2 installed successfully")
            return True
        except ImportError:
            print("❌ Failed to install pytube2")
            return False

def test_youtube_download():
    """Test downloading the specific YouTube video"""

    # The problematic URL
    video_url = "https://www.youtube.com/watch?v=KurGnK76-EU"

    print(f"🎯 Testing YouTube URL: {video_url}")
    print("=" * 60)

    try:
        from pytube import YouTube

        print("📡 Creating YouTube object...")
        yt = YouTube(video_url)

        print("📋 Getting video information...")
        try:
            title = yt.title
            print(f"📺 Title: {title}")
        except Exception as e:
            print(f"⚠️ Could not get title: {e}")
            title = "Unknown"

        try:
            duration = yt.length
            print(f"⏱️ Duration: {duration} seconds")
        except Exception as e:
            print(f"⚠️ Could not get duration: {e}")
            duration = 0

        try:
            views = yt.views
            print(f"👀 Views: {views}")
        except Exception as e:
            print(f"⚠️ Could not get views: {e}")
            views = 0

        print("\n🔍 Getting available streams...")
        streams = yt.streams

        print("📋 All available streams:")
        for i, stream in enumerate(streams, 1):
            print(f"   {i}. {stream}")

        print("\n🎬 Looking for progressive MP4 streams...")
        progressive_streams = yt.streams.filter(progressive=True, file_extension='mp4')

        if progressive_streams:
            print("✅ Progressive MP4 streams found:")
            for i, stream in enumerate(progressive_streams, 1):
                print(f"   {i}. {stream}")

            # Try to download the best one
            best_stream = progressive_streams.order_by('resolution').desc().first()
            print(f"\n📥 Attempting to download: {best_stream}")

            # Create downloads directory
            download_dir = "downloads"
            os.makedirs(download_dir, exist_ok=True)

            # Download
            print("⬇️ Starting download...")
            downloaded_file = best_stream.download(output_path=download_dir, filename="test_video.mp4")

            file_size = os.path.getsize(downloaded_file) / (1024 * 1024)  # MB
            print(f"✅ Download successful!")
            print(f"📁 File: {downloaded_file}")
            print(f"📊 Size: {file_size:.2f} MB")

        else:
            print("❌ No progressive MP4 streams found")
            print("🔍 Trying any MP4 streams...")

            mp4_streams = yt.streams.filter(file_extension='mp4')
            if mp4_streams:
                print("✅ MP4 streams found:")
                for i, stream in enumerate(mp4_streams, 1):
                    print(f"   {i}. {stream}")
            else:
                print("❌ No MP4 streams found at all")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

        # Check for specific error types
        if "403" in str(e):
            print("\n🚨 HTTP 403 Forbidden Error:")
            print("   - YouTube is blocking access to this video")
            print("   - Video might be age-restricted, private, or region-locked")
            print("   - Try a different video URL")

        elif "404" in str(e):
            print("\n🚨 HTTP 404 Not Found Error:")
            print("   - Video doesn't exist or has been removed")
            print("   - Check if the URL is correct")

        return False

    return True

def test_alternative_urls():
    """Test some alternative YouTube URLs that usually work"""

    alternative_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - Gangnam Style
        "https://youtu.be/dQw4w9WgXcQ"                  # Short URL format
    ]

    print("\n" + "=" * 60)
    print("🔄 Testing alternative YouTube URLs...")
    print("=" * 60)

    for i, url in enumerate(alternative_urls, 1):
        print(f"\n🎯 Test {i}: {url}")
        try:
            from pytube import YouTube
            yt = YouTube(url)
            title = yt.title
            duration = yt.length
            print(f"✅ Success: {title} ({duration}s)")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🐍 pytube2 Local Test Script")
    print("🎯 Testing YouTube Download Capabilities")
    print("=" * 60)

    # Install pytube2 if needed
    if not install_pytube2():
        print("❌ Cannot proceed without pytube2")
        sys.exit(1)

    # Test the specific problematic URL
    success = test_youtube_download()

    # If main URL fails, try alternatives
    if not success:
        test_alternative_urls()

    print("\n" + "=" * 60)
    print("📊 Test complete!")
    print("💡 If all URLs fail, pytube2 might have compatibility issues")
    print("🔧 Try: pip install --upgrade pytube2")
    print("🔧 Or try: pip install pytubefix (alternative)")
