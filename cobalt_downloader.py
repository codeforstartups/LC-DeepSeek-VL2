#!/usr/bin/env python3
"""
Simple YouTube Video Downloader using Cobalt.tools
Supports downloading videos from YouTube and other platforms
"""

import requests
import os
import sys
from urllib.parse import urlparse

class CobaltDownloader:
    def __init__(self):
        # Multiple API endpoints to try (fallback system)
        self.api_endpoints = [
            "https://cobalt-api.ayo.tf/api/json",
            "https://api.cobalt.tacohitbox.com/api/json",
            "https://cobalt.minaev.su/api/json",
            "https://cblt.fariz.dev/api/json",
            "https://co.wuk.sh/api/json"  # Original endpoint as fallback
        ]
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def download_video(self, video_url, download_audio_only=False):
        """
        Download video using Cobalt.tools API

        Args:
            video_url (str): URL of the video to download
            download_audio_only (bool): If True, download only audio

        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🎬 Starting download for: {video_url}")

        # Prepare request data
        payload = {
            "url": video_url,
            "vCodec": "h264",  # Video codec
            "vQuality": "720",  # Video quality
            "aFormat": "mp3",   # Audio format for audio-only downloads
            "filenamePattern": "basic",  # Simple filename
            "isAudioOnly": download_audio_only
        }

                # Try each API endpoint until one works
        for i, api_url in enumerate(self.api_endpoints):
            try:
                print(f"📡 Trying endpoint {i+1}/{len(self.api_endpoints)}: {api_url.split('/')[2]}...")
                response = requests.post(api_url, json=payload, headers=self.headers, timeout=15)

                if response.status_code != 200:
                    print(f"❌ Endpoint failed with status code: {response.status_code}")
                    if i < len(self.api_endpoints) - 1:
                        print("🔄 Trying next endpoint...")
                        continue
                    else:
                        print(f"Response: {response.text}")
                        return False

                data = response.json()
                print(f"✅ API Response from {api_url.split('/')[2]}: {data.get('status', 'unknown')}")
                break  # Success, exit the loop

            except requests.exceptions.RequestException as e:
                print(f"❌ Network error with {api_url.split('/')[2]}: {str(e)[:100]}...")
                if i < len(self.api_endpoints) - 1:
                    print("🔄 Trying next endpoint...")
                    continue
                else:
                    print("❌ All endpoints failed!")
                    return False
        else:
            print("❌ No working endpoints found!")
            return False

                # Handle different response types
        if data.get("status") == "error":
            print(f"❌ Error: {data.get('text', 'Unknown error')}")
            return False

        # Check for direct download URL
        if "url" in data:
            download_url = data["url"]
            filename = self._get_filename(data, video_url, download_audio_only)
            return self._download_file(download_url, filename)

        # Handle picker type (multiple files)
        elif data.get("status") == "picker":
            print("📂 Multiple files available:")
            for i, item in enumerate(data.get("picker", [])):
                print(f"   {i+1}. {item.get('url', 'No URL')}")

            # Download first file by default
            if data.get("picker"):
                download_url = data["picker"][0]["url"]
                filename = self._get_filename(data, video_url, download_audio_only, index=0)
                return self._download_file(download_url, filename)

        else:
            print(f"❌ Unexpected response format: {data}")
            return False

    def _get_filename(self, data, original_url, is_audio=False, index=0):
        """Generate filename for downloaded file"""
        # Try to get filename from response
        if "filename" in data:
            return data["filename"]

        # Generate filename from URL
        parsed_url = urlparse(original_url)
        if "youtube.com/watch" in original_url or "youtu.be" in original_url:
            video_id = original_url.split("v=")[-1].split("&")[0] if "v=" in original_url else parsed_url.path[1:]
            if len(video_id) > 11:  # YouTube video IDs are 11 characters
                video_id = video_id[:11]

            extension = "mp3" if is_audio else "mp4"
            return f"youtube_{video_id}.{extension}"

        # Fallback filename
        extension = "mp3" if is_audio else "mp4"
        return f"downloaded_video_{index}.{extension}"

    def _download_file(self, url, filename):
        """Download file from URL"""
        try:
            print(f"⬇️  Downloading: {filename}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size
            total_size = int(response.headers.get('content-length', 0))

            with open(filename, 'wb') as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        # Show progress
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📥 Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")

            print(f"\n✅ Successfully downloaded: {filename}")
            print(f"📁 File saved in: {os.path.abspath(filename)}")
            return True

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False

def main():
    """Main function"""
    print("🚀 Cobalt.tools YouTube Downloader")
    print("=" * 40)

    downloader = CobaltDownloader()

    # Default video URL (the one you specified)
    default_url = "https://www.youtube.com/watch?v=KurGnK76-EU"

    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        print(f"Using default video: {default_url}")
        video_url = default_url

    # Ask user for download type
    print("\nChoose download type:")
    print("1. Video (MP4)")
    print("2. Audio only (MP3)")

    try:
        choice = input("Enter your choice (1 or 2, default is 1): ").strip()
        download_audio = choice == "2"
    except KeyboardInterrupt:
        print("\n👋 Download cancelled by user")
        sys.exit(0)

    # Start download
    print("\n" + "="*40)
    success = downloader.download_video(video_url, download_audio_only=download_audio)

    if success:
        print("\n🎉 Download completed successfully!")
    else:
        print("\n😞 Download failed. Please try again or check the URL.")

if __name__ == "__main__":
    main()
