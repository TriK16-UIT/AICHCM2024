import whisperx
import gc 
import json
import os
import json
from transformers import pipeline
from moviepy.editor import *
from IPython.display import Audio
from pprint import pprint
import torch
from tqdm import tqdm
import sys
import re

def get_video_number(filename):
    match = re.search(r'L(\d+)', filename)
    return int(match.group(1)) if match else 0

def start_from_index(dir, index, format):
    files = [f for f in os.listdir(dir) if f.endswith(format)]
    files.sort(key=get_video_number)
    files_to_process = [f for f in files if get_video_number(f) >= index]

    return files_to_process

sys.path.append('..')
from config import AUDIO_DETECTION_PATH, AUDIO_PATH, VIDEOS_PATH
video_dir = "../" + VIDEOS_PATH
audio_dir = "../" + AUDIO_PATH
audio_detection_dir = "../" + AUDIO_DETECTION_PATH

video_files_to_process = start_from_index(video_dir, 25, "")

print(video_files_to_process[0])
print(len(video_files_to_process))

def preprocess_text(text: str):
    text = text.lower()
    reg_pattern = r'[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸÝửữựỳỵỷỹý\s]'
    output = re.sub(reg_pattern, '', text)
    output = output.strip()
    output = " ".join(output.split())
    return output

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = whisperx.load_model("large-v2", device, compute_type="float16")

batch_size=8
for video_filename in tqdm(video_files_to_process):
    video_path = os.path.join(video_dir, video_filename)
    video = VideoFileClip(video_path)
    audio_path = os.path.join(audio_dir, video_filename).replace("mp4", "wav")
    video.audio.write_audiofile(audio_path)

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    if os.path.isfile(audio_path):
        os.remove(audio_path)

    timestamps = [[segment['start'], segment['end']] for segment in result['segments']]
    texts = [preprocess_text(segment['text']) for segment in result['segments']]

    output_filename = video_filename.replace(".mp4", ".json")
    timestamps_path = os.path.join(audio_detection_dir, output_filename)
    texts_path = os.path.join(audio_dir, output_filename)

    with open(timestamps_path, 'w') as f:
        json.dump(timestamps, f)
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False)

    print(f"Completed on {video_filename}")


