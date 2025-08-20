import cv2
import os
import argparse
from ultralytics import YOLO

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="Input images directory")
parser.add_argument("output_dir", help="Output images directory")
parser.add_argument("person_det_model", help="Path to person detection model weights")
parser.add_argument("ppe_det_model", help="Path to PPE detection model weights")
args = parser.parse_args()

# Load models
person_model = YOLO(args.person_det_model)
ppe_model = YOLO(args.ppe_det_model)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Process each image
for file_name in os.listdir(args.input_dir):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(args.input_dir, file_name)
        image = cv2.imread(img_path)

        # Run person detection
        person_results = person_model(img_path)[0]
        # Run PPE detection
        ppe_results = ppe_model(img_path)[0]

        # Draw person detections
        for box in person_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = "Person"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw PPE detections
        for box in ppe_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = ppe_model.names[cls_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Save annotated image
        out_path = os.path.join(args.output_dir, file_name)
        cv2.imwrite(out_path, image)
        print(f"Processed & saved: {out_path}")

print("✅ Inference completed. Check output images.")
