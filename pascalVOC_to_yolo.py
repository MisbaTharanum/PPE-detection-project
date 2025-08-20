#!/usr/bin/env python3
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO TXT format")
    parser.add_argument("input_dir", help="Base input directory containing images and XML annotations")
    parser.add_argument("output_dir", help="Output directory for YOLO-format images/labels")
    return parser.parse_args()

def find_image(filename, search_dir):
    """Search for image file with given filename in all subfolders."""
    for p in Path(search_dir).rglob(filename):
        if p.is_file():
            return p
    return None

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    class_names = []
    xml_files = list(input_dir.rglob("*.xml"))

    if not xml_files:
        print("❌ No XML annotation files found in", input_dir)
        return

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text.strip()
        img_path = find_image(filename, input_dir)

        if img_path is None:
            print(f"❌ Image for {filename} not found, skipping {xml_file.name}")
            continue

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in class_names:
                class_names.append(cls_name)
            cls_id = class_names.index(cls_name)

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_file = labels_out / (xml_file.stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        shutil.copy(img_path, images_out / img_path.name)

    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(class_names))

    print("✅ Conversion complete.")
    print(f"Classes found: {class_names}")
    print(f"Images saved to: {images_out}")
    print(f"Labels saved to: {labels_out}")

if __name__ == "__main__":
    main()
