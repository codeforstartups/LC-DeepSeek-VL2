import requests
import json

# Your API endpoint
url = "http://13.59.72.219:8004/match_faces"

# Your payload
payload = {
    "video_url": "https://vision-test-deepseekvl2.s3.us-east-2.amazonaws.com/temp.mp4",
    "photos_bucket": "vision-test-deepseekvl2",
    "photos_prefix": "faces_photos/"
}

print("Testing face recognition API...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success! Found {len(result['matches'])} matches:")
        for i, match in enumerate(result['matches']):
            print(f"  Match {i+1}:")
            print(f"    Photo: {match['photo_key']}")
            print(f"    Frame: {match['frame_index']}")
            print(f"    Timestamp: {match['timestamp_ms']}ms")
    else:
        print(f"Error: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
