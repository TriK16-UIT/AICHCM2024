from ultralytics import YOLOv10
import cv2
import os
from collections import defaultdict
import json
from tqdm import tqdm
import re

def get_video_number(filename):
    match = re.search(r'L(\d+)', filename)
    return int(match.group(1)) if match else 0

def start_from_index(dir, index, format):
    files = [f for f in os.listdir(dir) if f.endswith(format)]
    files.sort(key=get_video_number)
    files_to_process = [f for f in files if get_video_number(f) >= index]

    return files_to_process

model = YOLOv10('pretrained/yolov10l.pt')
idx = 0
object_dict = defaultdict(lambda: defaultdict(lambda: {"count": 0}))

keyframes_dir = 'E:\AICHCM2024\Data\improved_keyframes'

json_path = 'E:\AICHCM2024\DataPreprocessing\object.json'
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        object_dict = defaultdict(lambda: defaultdict(lambda: {"count": 0}), json.load(f))
    start_idx = max(map(int, object_dict.keys())) + 1 if object_dict else 0
else:
    start_idx = 0

print("Current index: ", start_idx)

keyframe_files_to_process = start_from_index(keyframes_dir, 25, "")

print(keyframe_files_to_process[0])
print(len(keyframe_files_to_process))

for video_folder in keyframe_files_to_process:
    video_folder_path = os.path.join(keyframes_dir, video_folder)

    image_paths = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(img_path)
        results = model.predict(img, conf=0.20, device="cuda:0")
        
        detected = False 

        for result in results:
            for box in result.boxes:
                object_dict[start_idx][result.names[int(box.cls[0])]]["count"] += 1
                detected = True

        if not detected:
            object_dict[start_idx] = {}

        start_idx += 1

with open('E:\AICHCM2024\DataPreprocessing\object.json', 'w') as f:
    json.dump(object_dict, f, indent=4)

print(f"Processing complete. Results saved to {json_path}")