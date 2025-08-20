1. Project Title & Overview

Briefly describe the project goal:

“Two-stage YOLOv8 pipeline for Person Detection and PPE Detection. Model 1 detects persons, Model 2 detects PPE items in cropped person images.”

Mention the PPE classes trained (hard-hat, gloves, mask, glasses, boots, vest).

2. Folder Structure

Example:

PPE_PROJECT/
├─ README.md
├─ report/PPE_Detection_Report.pdf
├─ scripts/
│  ├─ pascalVOC_to_yolo.py
│  ├─ inference.py
│  ├─ checkingmodel.py
├─ configs/
│  ├─ person_data.yaml
│  └─ ppe_data.yaml
├─ weights/
│  ├─ person_det_model.pt
│  └─ ppe_det_model.pt
├─ examples/
│  ├─ input/
│  └─ output/
└─ results/
   ├─ person_val/
   └─ ppe_val/

3. Requirements

Python version (3.9+)

Libraries:

pip install ultralytics opencv-python

4. Steps to Run

Step 1 – Annotation Conversion (Q1)

python scripts/pascalVOC_to_yolo.py --input_dir path/to/VOC/annotations --output_dir path/to/output_labels


Step 2 – Train Person Detection Model (Q2)

yolo task=detect mode=train model=yolov8n.pt data=configs/person_data.yaml epochs=50 imgsz=640


Step 3 – Train PPE Detection Model (Q3)

yolo task=detect mode=train model=yolov8n.pt data=configs/ppe_data.yaml epochs=50 imgsz=640


Step 4 – Validate Models

yolo task=detect mode=val model=weights/person_det_model.pt data=configs/person_data.yaml
yolo task=detect mode=val model=weights/ppe_det_model.pt data=configs/ppe_data.yaml


Step 5 – Inference Pipeline (Q4 & Q5)

python scripts/inference.py examples/input examples/output weights/person_det_model.pt weights/ppe_det_model.pt

5. Results

Mention validation mAP, precision, recall for both models.

Add confusion matrix images for PPE model.

6. Key Learnings

What you learned while implementing (dataset imbalance, class filtering, OpenCV drawing, etc.)