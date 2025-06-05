import json
import re

def parse_detection_response(response_text):
    """Parse the structured response from DeepSeek"""

    # Initialize result
    result = {
        "answer": None,
        "description": "",
        "confidence": None
    }

    # Split by lines and process each
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Extract Answer (YES/NO)
        if line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            result["answer"] = answer.upper() == "YES"

        # Extract Description
        elif line.startswith("Description:"):
            result["description"] = line.replace("Description:", "").strip()

        # Extract Confidence (try to get just the number)
        elif line.startswith("Confidence:"):
            conf_text = line.replace("Confidence:", "").strip()
            # Try to extract number (1-10)
            numbers = re.findall(r'\d+', conf_text)
            if numbers:
                result["confidence"] = int(numbers[0])

    return result

def analyze_video_detections(json_data):
    """Analyze all frame detections and extract key insights"""

    detections = []
    found_frames = []

    for frame in json_data["frame_responses"]:
        second = frame["second"]
        raw_response = frame["deepseek_response"]

        # Parse the response
        parsed = parse_detection_response(raw_response)

        detection_result = {
            "second": second,
            "found": parsed["answer"],
            "description": parsed["description"],
            "confidence": parsed["confidence"]
        }

        detections.append(detection_result)

        # Track frames where object was found
        if parsed["answer"]:
            found_frames.append(second)

    # Summary statistics
    total_frames = len(detections)
    positive_detections = len(found_frames)

    summary = {
        "query": json_data["query"],
        "video_duration": json_data["video_duration"],
        "total_frames": total_frames,
        "positive_detections": positive_detections,
        "detection_rate": f"{(positive_detections/total_frames)*100:.1f}%",
        "found_at_seconds": found_frames,
        "detections": detections
    }

    return summary

# Example usage
if __name__ == "__main__":
    # Your JSON response
    response_data = {
        "query": "find remote",
        "video_duration": 21.27,
        "total_frames_analyzed": 22,
        "frame_responses": [
            {
                "second": 8,
                "frame_file": "frame_008.jpg",
                "deepseek_response": "Answer: Yes\n\nDescription: The image shows a wooden table with a black remote control placed on it. In front of the table is an electronic circuit board partially visible at the bottom edge of the frame.\n\nConfidence: 8"
            },
            {
                "second": 16,
                "frame_file": "frame_016.jpg",
                "deepseek_response": "Answer: NO\n\nDescription: The image shows an indoor setting with a dining area featuring a rectangular table surrounded by six chairs.\n\nConfidence: 5"
            }
        ]
    }

    # Parse and analyze
    results = analyze_video_detections(response_data)

    print("📊 DETECTION ANALYSIS")
    print("=" * 50)
    print(f"Query: {results['query']}")
    print(f"Video Duration: {results['video_duration']}s")
    print(f"Detection Rate: {results['detection_rate']}")
    print(f"Found at seconds: {results['found_at_seconds']}")
    print()

    print("🔍 FRAME-BY-FRAME RESULTS")
    print("=" * 50)
    for detection in results['detections']:
        status = "✅ FOUND" if detection['found'] else "❌ NOT FOUND"
        conf = detection['confidence'] or "N/A"
        print(f"Second {detection['second']:2d}: {status} (Confidence: {conf})")
        if detection['found']:
            print(f"          Description: {detection['description'][:80]}...")
        print()