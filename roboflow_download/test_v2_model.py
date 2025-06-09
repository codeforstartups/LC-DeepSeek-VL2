#!/usr/bin/env python3
"""
Test the Enhanced V2 Thermal Gun Detection Model
Compare performance with V1 and validate improvements
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob
from datetime import datetime

def test_v2_model():
    """Test the new V2 improved model"""
    print("🔍 Testing Enhanced V2 Thermal Gun Detection Model")
    print("=" * 60)

    # Model paths
    v2_model_path = "../MyTrainedModels/ThermalPistol_v2_improved/weights/best.pt"
    v1_model_path = "../MyTrainedModels/ThermalPistol_v1/weights/best.pt"

    # Check if models exist
    if not os.path.exists(v2_model_path):
        print(f"❌ V2 model not found at: {v2_model_path}")
        return

    print(f"✅ V2 model found: {v2_model_path}")

    # Load V2 model
    print("📦 Loading V2 model...")
    model_v2 = YOLO(v2_model_path)

    # Test on validation images
    val_images_dir = "valid/images"
    image_files = glob.glob(os.path.join(val_images_dir, "*.jpg"))

    print(f"📊 Found {len(image_files)} validation images")

    # Test different confidence thresholds
    confidence_thresholds = [0.1, 0.25, 0.5, 0.75]

    print(f"\n🎯 Testing V2 Model Performance:")
    print("-" * 40)

    for conf in confidence_thresholds:
        total_detections = 0
        images_with_detections = 0

        for img_path in image_files:
            # Run inference
            results = model_v2(img_path, conf=conf, verbose=False)

            # Count detections
            detections = 0
            if len(results) > 0 and results[0].boxes is not None:
                detections = len(results[0].boxes)

            total_detections += detections
            if detections > 0:
                images_with_detections += 1

        print(f"Confidence {conf:>4}: {images_with_detections:>2}/{len(image_files)} images, {total_detections:>3} total detections")

    # Test on a few sample images with visualization
    print(f"\n🖼️  Detailed Results on Sample Images:")
    print("-" * 40)

    sample_images = image_files[:5]  # Test first 5 images
    results_dir = "v2_test_results"
    os.makedirs(results_dir, exist_ok=True)

    for i, img_path in enumerate(sample_images):
        img_name = os.path.basename(img_path)
        print(f"\n📸 {img_name}:")

        # Load image
        img = cv2.imread(img_path)

        # Test with confidence 0.25
        results = model_v2(img_path, conf=0.25, verbose=False)

        # Create visualization
        vis_img = img.copy()
        detections = 0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detections = len(boxes)

            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                confidence = boxes.conf[j].cpu().numpy()

                # Draw bounding box
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_img, f'Gun {confidence:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"   Detection {j+1}: confidence={confidence:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

        print(f"   Total detections: {detections}")

        # Save visualization
        output_path = os.path.join(results_dir, f"v2_test_{i+1:02d}_{img_name}")
        cv2.imwrite(output_path, vis_img)
        print(f"   Saved: {output_path}")

    print(f"\n📁 Visualizations saved to: {results_dir}/")

    # Compare with V1 if available
    if os.path.exists(v1_model_path):
        print(f"\n🔄 Comparing with V1 Model...")
        print("-" * 40)

        model_v1 = YOLO(v1_model_path)

        # Test V1 on same sample images
        v1_detections = 0
        v2_detections = 0

        for img_path in sample_images[:3]:  # Test 3 images
            # V1 results
            results_v1 = model_v1(img_path, conf=0.25, verbose=False)
            v1_det = len(results_v1[0].boxes) if len(results_v1) > 0 and results_v1[0].boxes is not None else 0

            # V2 results
            results_v2 = model_v2(img_path, conf=0.25, verbose=False)
            v2_det = len(results_v2[0].boxes) if len(results_v2) > 0 and results_v2[0].boxes is not None else 0

            v1_detections += v1_det
            v2_detections += v2_det

            img_name = os.path.basename(img_path)
            print(f"   {img_name}: V1={v1_det}, V2={v2_det}")

        print(f"\n📊 Model Comparison (3 sample images):")
        print(f"   V1 Total Detections: {v1_detections}")
        print(f"   V2 Total Detections: {v2_detections}")
        print(f"   Improvement: {'+' if v2_detections >= v1_detections else ''}{v2_detections - v1_detections} detections")

    # Final recommendations
    print(f"\n💡 V2 Model Assessment:")
    print("-" * 40)

    # Run validation on the dataset
    print("🔍 Running full validation...")
    try:
        val_results = model_v2.val(data="data_fixed.yaml", split='val')
        precision = val_results.box.p.mean()
        recall = val_results.box.r.mean()
        map50 = val_results.box.map50
        map50_95 = val_results.box.map

        print(f"✅ Validation Results:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   mAP@0.5: {map50:.3f}")
        print(f"   mAP@0.5-0.95: {map50_95:.3f}")

        # Assessment
        if map50 > 0.8:
            print(f"🎉 EXCELLENT! Model performs very well (mAP@0.5 > 0.8)")
        elif map50 > 0.5:
            print(f"✅ GOOD! Model performs adequately (mAP@0.5 > 0.5)")
        elif map50 > 0.2:
            print(f"⚠️  FAIR! Model needs improvement (mAP@0.5 > 0.2)")
        else:
            print(f"❌ POOR! Model needs major work (mAP@0.5 < 0.2)")

    except Exception as e:
        print(f"❌ Validation failed: {e}")

    print(f"\n🎯 Ready for Real-World Testing!")
    print(f"Next step: Test on your thermal video with the command:")
    print(f"   python test_thermal_video_v2.py")

if __name__ == "__main__":
    test_v2_model()
