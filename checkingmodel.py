import os
from collections import Counter

# Change this to your labels/train or labels/val path
label_paths = [
    r"C:/Users/Shifa/Desktop/datasets/labels/train",
    r"C:/Users/Shifa/Desktop/datasets/labels/val"
]

# Classes as per your dataset.yaml
class_names = [
    "hard-hat",
    "gloves",
    "mask",
    "glasses",
    "boots",
    "vest",
    "ppe-suit",
    "ear-protector",
    "safety-harness"
]


counter = Counter()

# Count class occurrences
for path in label_paths:
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

# Print results
print("Class distribution:")
for class_id, count in counter.items():
    print(f"{class_names[class_id]} ({class_id}): {count} instances")


