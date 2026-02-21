# check_detection_classes.py
import json
from collections import Counter

json_path = "data/bdd100k/labels/detection/bdd100k_labels_images_train.json"

with open(json_path, "r") as f:
    data = json.load(f)

counter = Counter()

for item in data:
    for obj in item.get("labels", []):
        if "category" in obj:
            counter[obj["category"]] += 1

print("Detection classes:")
for cls, cnt in counter.most_common():
    print(f"{cls:15s}: {cnt}")
