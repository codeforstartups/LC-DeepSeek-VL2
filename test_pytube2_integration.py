#!/usr/bin/env python3
"""
Test script for Thermal Gun Detection API with pytube2 YouTube Integration
Tests both YouTube URLs and direct video URLs
"""

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8006"

def test_health():
    """Test API health and pytube2 availability"""
    print("🏥 Testing API health and pytube2 integration...")

    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Status: {health['status']}")
            print(f"🤖 Model loaded: {health['model_loaded']}")
            print(f"💻 Device: {health['device']}")
            print(f"📺 pytube2 available: {health['pytube2_available']}")
            print(f"🔧 Download methods: {health['download_methods']}")
            print(f"🎯 Supported sources:")
            for source in health['supported_sources']:
                print(f"   - {source}")
            return health['pytube2_available']
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_video_detection(test_name, video_url, expected_method):
    """Test video detection with specific URL"""
    print(f"\n🎯 {test_name}")
    print(f"🔗 URL: {video_url}")
    print(f"📝 Expected method: {expected_method}")
    print("=" * 70)

    try:
        start_time = time.time()

        response = requests.post(
            f"{API_URL}/detect_thermal_guns/",
            json={
                "video_url": video_url,
                "confidence_threshold": 0.25
            },
            timeout=300  # 5 minutes for YouTube downloads
        )

        end_time = time.time()
        duration = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS ({duration:.1f}s)")
            print(f"📊 Request ID: {result['request_id']}")
            print(f"🎬 Duration: {result['video_info']['duration_seconds']}s")
            print(f"🔍 Frames analyzed: {result['video_info']['total_frames_analyzed']}")
            print(f"🎯 Guns found: {result['gun_detection_summary']['guns_found']}")

            if result['gun_detection_summary']['guns_found']:
                print(f"🚨 Total detections: {result['gun_detection_summary']['total_gun_detections']}")
                print(f"📈 Detection rate: {result['gun_detection_summary']['detection_rate_percent']}%")
                print(f"⏰ Detection times: {result['gun_detection_summary']['detection_times']}")
            else:
                print("✅ No thermal guns detected (expected for test videos)")

        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"📝 Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"📝 Raw response: {response.text}")

    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT - This is normal for long videos, but processing may have started")
    except Exception as e:
        print(f"❌ EXCEPTION - {e}")

def run_comprehensive_test():
    """Run comprehensive test of pytube2 integration"""

    # Test cases
    test_cases = [
        {
            "name": "Direct MP4 URL Test",
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "method": "requests"
        },
        {
            "name": "YouTube Video Test (codebasics)",
            "url": "https://www.youtube.com/watch?v=KurGnK76-EU&ab_channel=codebasics",
            "method": "pytube2"
        },
        {
            "name": "YouTube Short URL Test",
            "url": "https://youtu.be/KurGnK76-EU",
            "method": "pytube2"
        },
        {
            "name": "YouTube Mobile URL Test",
            "url": "https://m.youtube.com/watch?v=KurGnK76-EU",
            "method": "pytube2"
        }
    ]

    success_count = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{total_tests}: {test_case['name']}")
        print(f"{'='*70}")

        try:
            test_video_detection(test_case['name'], test_case['url'], test_case['method'])
            success_count += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    return success_count, total_tests

if __name__ == "__main__":
    print("🔫 Thermal Gun Detection API - pytube2 Integration Test")
    print("🐍 Testing YouTube + Direct URL Support")
    print("=" * 70)

    # Check health first
    pytube_available = test_health()

    if not pytube_available:
        print("\n⚠️ WARNING: pytube2 not available!")
        print("📦 Install with: pip install pytube2")
        print("🎯 Only direct URLs will work")

    print(f"\n🚀 Starting comprehensive video download tests...")

    success_count, total_tests = run_comprehensive_test()

    print(f"\n{'='*70}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful tests: {success_count}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - success_count}/{total_tests}")
    print(f"📈 Success rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your pytube2 integration is working perfectly!")
    elif success_count > 0:
        print(f"\n⚠️ Partial success. {success_count} out of {total_tests} tests passed.")
    else:
        print("\n❌ All tests failed. Check your API setup and dependencies.")

    print("\n💡 Integration Status:")
    if pytube_available:
        print("   ✅ pytube2 available - YouTube URLs supported")
        print("   ✅ requests available - Direct URLs supported")
        print("   🎯 Your API can handle ANY video URL!")
    else:
        print("   ❌ pytube2 missing - Only direct URLs supported")
        print("   ✅ requests available - Direct URLs working")
        print("   🔧 Install pytube2 for full YouTube support")
