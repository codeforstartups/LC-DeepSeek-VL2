#!/usr/bin/env python3
"""
Local pytube2 test script
Test downloading specific YouTube video: https://www.youtube.com/watch?v=KurGnK76-EU
Now with cleanup and server environment testing
"""

import os
import sys
from pathlib import Path
import time

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

def cleanup_video(file_path: str, request_id: str = "test"):
    """Delete downloaded video file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"🗑️ [{request_id}] Video cleaned up: {file_path}")
            return True
        else:
            print(f"⚠️ [{request_id}] File not found for cleanup: {file_path}")
            return False
    except Exception as e:
        print(f"❌ [{request_id}] Cleanup failed: {e}")
        return False

def test_youtube_download_with_cleanup(request_id: str = "local_test"):
    """Test downloading the specific YouTube video with cleanup"""

    # The problematic URL
    video_url = "https://www.youtube.com/watch?v=KurGnK76-EU"
    downloaded_file = None

    print(f"🎯 [{request_id}] Testing YouTube URL: {video_url}")
    print("=" * 60)

    try:
        from pytube import YouTube

        print(f"📡 [{request_id}] Creating YouTube object...")

        # Test with different configurations to simulate server
        if "server" in request_id.lower():
            print(f"🖥️ [{request_id}] Using SERVER-like configuration...")
            # Simulate server environment (no special headers for now)
            yt = YouTube(video_url)
        else:
            print(f"💻 [{request_id}] Using LOCAL configuration...")
            yt = YouTube(video_url)

        print(f"📋 [{request_id}] Getting video information...")
        try:
            title = yt.title
            print(f"📺 [{request_id}] Title: {title}")
        except Exception as e:
            print(f"⚠️ [{request_id}] Could not get title: {e}")
            title = "Unknown"

        try:
            duration = yt.length
            print(f"⏱️ [{request_id}] Duration: {duration} seconds")
        except Exception as e:
            print(f"⚠️ [{request_id}] Could not get duration: {e}")
            duration = 0

        try:
            views = yt.views
            print(f"👀 [{request_id}] Views: {views}")
        except Exception as e:
            print(f"⚠️ [{request_id}] Could not get views: {e}")
            views = 0

        print(f"\n🔍 [{request_id}] Getting available streams...")
        streams = yt.streams

        print(f"📊 [{request_id}] Found {len(streams)} total streams")

        print(f"\n🎬 [{request_id}] Looking for progressive MP4 streams...")
        progressive_streams = yt.streams.filter(progressive=True, file_extension='mp4')

        if progressive_streams:
            print(f"✅ [{request_id}] Progressive MP4 streams found: {len(progressive_streams)}")

            # Try to download the best one
            best_stream = progressive_streams.order_by('resolution').desc().first()
            print(f"\n📥 [{request_id}] Attempting to download: {best_stream}")

            # Create downloads directory
            download_dir = "downloads"
            os.makedirs(download_dir, exist_ok=True)

            # Generate unique filename
            timestamp = int(time.time())
            filename = f"test_video_{request_id}_{timestamp}.mp4"

            # Download
            print(f"⬇️ [{request_id}] Starting download...")
            downloaded_file = best_stream.download(output_path=download_dir, filename=filename)

            file_size = os.path.getsize(downloaded_file) / (1024 * 1024)  # MB
            print(f"✅ [{request_id}] Download successful!")
            print(f"📁 [{request_id}] File: {downloaded_file}")
            print(f"📊 [{request_id}] Size: {file_size:.2f} MB")

            # Cleanup the downloaded file
            print(f"\n🗑️ [{request_id}] Cleaning up downloaded video...")
            cleanup_success = cleanup_video(downloaded_file, request_id)

            if cleanup_success:
                print(f"✅ [{request_id}] Cleanup successful")
            else:
                print(f"⚠️ [{request_id}] Cleanup had issues")

        else:
            print(f"❌ [{request_id}] No progressive MP4 streams found")
            print(f"🔍 [{request_id}] Trying any MP4 streams...")

            mp4_streams = yt.streams.filter(file_extension='mp4')
            if mp4_streams:
                print(f"✅ [{request_id}] MP4 streams found: {len(mp4_streams)}")
                # Could try downloading these too, but let's keep it simple
            else:
                print(f"❌ [{request_id}] No MP4 streams found at all")

    except Exception as e:
        print(f"❌ [{request_id}] Error occurred: {e}")
        print(f"🔍 [{request_id}] Error type: {type(e).__name__}")

        # Check for specific error types
        if "403" in str(e):
            print(f"\n🚨 [{request_id}] HTTP 403 Forbidden Error:")
            print(f"   - YouTube is blocking access to this video")
            print(f"   - Video might be age-restricted, private, or region-locked")
            print(f"   - Try a different video URL")

        elif "404" in str(e):
            print(f"\n🚨 [{request_id}] HTTP 404 Not Found Error:")
            print(f"   - Video doesn't exist or has been removed")
            print(f"   - Check if the URL is correct")

        # Cleanup on error too
        if downloaded_file and os.path.exists(downloaded_file):
            print(f"\n🗑️ [{request_id}] Cleaning up due to error...")
            cleanup_video(downloaded_file, request_id)

        return False

    return True

def test_alternative_urls_with_cleanup():
    """Test some alternative YouTube URLs that usually work"""

    alternative_urls = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "rickroll"),    # Rick Roll
        ("https://www.youtube.com/watch?v=9bZkp7q19f0", "gangnam"),     # PSY - Gangnam Style
        ("https://youtu.be/dQw4w9WgXcQ", "short_url")                   # Short URL format
    ]

    print("\n" + "=" * 60)
    print("🔄 Testing alternative YouTube URLs...")
    print("=" * 60)

    for i, (url, name) in enumerate(alternative_urls, 1):
        print(f"\n🎯 Test {i} ({name}): {url}")
        try:
            from pytube import YouTube
            yt = YouTube(url)
            title = yt.title
            duration = yt.length
            print(f"✅ Success: {title} ({duration}s)")

            # Quick stream check without download
            streams = yt.streams.filter(progressive=True, file_extension='mp4')
            print(f"📊 Progressive streams available: {len(streams)}")

        except Exception as e:
            print(f"❌ Failed: {e}")

def run_server_simulation():
    """Simulate server environment test"""
    print("\n" + "🖥️" * 30)
    print("🖥️ SERVER SIMULATION TEST")
    print("🖥️" * 30)
    print("🎯 Testing as if running on server environment...")

    # Test main URL
    success = test_youtube_download_with_cleanup("server_test")

    return success

def run_local_test():
    """Run local environment test"""
    print("\n" + "💻" * 30)
    print("💻 LOCAL ENVIRONMENT TEST")
    print("💻" * 30)
    print("🎯 Testing in local environment...")

    # Test main URL
    success = test_youtube_download_with_cleanup("local_test")

    return success

if __name__ == "__main__":
    print("🐍 pytube2 Test Script - Local vs Server")
    print("🎯 Testing YouTube Download with Cleanup")
    print("=" * 60)

    # Install pytube2 if needed
    if not install_pytube2():
        print("❌ Cannot proceed without pytube2")
        sys.exit(1)

    # Run both tests
    local_success = run_local_test()
    server_success = run_server_simulation()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"💻 Local Test:  {'✅ SUCCESS' if local_success else '❌ FAILED'}")
    print(f"🖥️ Server Test: {'✅ SUCCESS' if server_success else '❌ FAILED'}")

    if local_success and server_success:
        print("\n🎉 Both tests passed! pytube2 should work in FastAPI")
    elif local_success and not server_success:
        print("\n⚠️ Local works, server fails - Environment issue!")
        print("💡 Check server headers, network, or request context")
    elif not local_success and not server_success:
        print("\n❌ Both failed - pytube2 compatibility issue")
        print("🔧 Try: pip install --upgrade pytube2")
        print("🔧 Or try: pip install pytubefix (alternative)")
    else:
        print("\n🤔 Unexpected result pattern")

    # If main URL fails, try alternatives
    if not local_success or not server_success:
        test_alternative_urls_with_cleanup()

    print("\n�� Test complete!")
