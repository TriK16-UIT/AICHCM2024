import sys
import os
import json
from tqdm import tqdm
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import re

def get_video_number(filename):
    match = re.search(r'L(\d+)', filename)
    return int(match.group(1)) if match else 0

def start_from_index(dir, index, format):
    files = [f for f in os.listdir(dir) if f.endswith(format)]
    files.sort(key=get_video_number)
    files_to_process = [f for f in files if get_video_number(f) >= index]

    return files_to_process

sys.path.append('../..')

keyframes_dir = "../../Data/improved_keyframes"
ocr_dir = "../../Data/ocr"

keyframe_files_to_process = start_from_index(keyframes_dir, 25, "")

print(keyframe_files_to_process[0])
print(len(keyframe_files_to_process))

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=True, rec=False, gpu_mem=4000)

config = Cfg.load_config_from_name('vgg_seq2seq')
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
viet_ocr = Predictor(config)

def crop_image(image_path, box):
    image = Image.open(image_path)
    left, top, right, bottom = (
        min(coord[0] for coord in box),
        min(coord[1] for coord in box),
        max(coord[0] for coord in box),
        max(coord[1] for coord in box)
    )
    return image.crop((left, top, right, bottom))

def process_video_folder(video_folder_path):
    image_paths = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    video_ocr_results = []

    for img_path in image_paths:
        detected_boxes = paddle_ocr.ocr(img_path, cls=True, rec=False)
        
        detected_boxes = detected_boxes[0]
        if not detected_boxes:
            video_ocr_results.append([])
        else:
            cropped_images = [crop_image(img_path, box) for box in detected_boxes]

            batch_size = 128
            frame_results = []
            for i in range(0, len(cropped_images), batch_size):
                batch = cropped_images[i:i+batch_size]
                texts, probs = viet_ocr.predict_batch(batch, return_prob=True)

                # Filter results based on probability
                filtered_results = [text for text, prob in zip(texts, probs) if prob >= 0.5]
                frame_results.extend(filtered_results)

            video_ocr_results.append(frame_results)
    
    return video_ocr_results

if not os.path.exists(ocr_dir):
    os.makedirs(ocr_dir)

for video_folder in tqdm(keyframe_files_to_process):
    video_folder_path = os.path.join(keyframes_dir, video_folder)

    ocr_results = process_video_folder(video_folder_path)

    json_filename = f"{video_folder}.json"
    json_path = os.path.join(ocr_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, ensure_ascii=False)  

print("All video folders processed.")
