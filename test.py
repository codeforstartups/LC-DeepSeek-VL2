import requests
import sys

def test_yolo_api(video_path):
    url = "http://3.19.70.22:8001/detect_objects/"

    try:
        with open(video_path, "rb") as video_file:
            files = {"video": video_file}
            print(f"Sending {video_path} to YOLO API...")
            response = requests.post(url, files=files)

            if response.status_code == 200:
                print("\nDetection Results:")
                print(response.json())
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on port 8001")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_video>")
        print("Example: python test.py test.mp4")
        sys.exit(1)

    test_yolo_api(sys.argv[1])