import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import os
import glob
import re

def get_video_number(filename):
    match = re.search(r'L(\d+)', filename)
    return int(match.group(1)) if match else 0

def start_from_index(dir, index, format):
    files = [f for f in os.listdir(dir) if f.endswith(format)]
    files.sort(key=get_video_number)
    files_to_process = [f for f in files if get_video_number(f) >= index]

    return files_to_process

def save_to_csv(output_file, lst_video_id, lst_frame_idx):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for video_id, frame_idx in zip(lst_video_id, lst_frame_idx):
            writer.writerow([video_id, frame_idx])

def extract_video_id_and_info(keyframe_path, keyframeMapper):
    parts = keyframe_path.split(os.sep)
    video_id = parts[-2]
    keyframe_id = os.path.splitext(parts[-1])[0]

    frame_idx = keyframeMapper.get(video_id, {}).get(str(int(keyframe_id)), {}).get('frame_idx', 'N/A')
    pts_time = keyframeMapper.get(video_id, {}).get(str(int(keyframe_id)), {}).get('pts_time', 'N/A')
   
    return video_id, frame_idx, pts_time

def get_nearby_frames(keyframe_path, n):
    ## This function will look for n nearby frames before and after the chosen frame    
    base_dir = os.path.dirname(keyframe_path)
    keyframe_id = int(os.path.splitext(os.path.basename(keyframe_path))[0])
    prefix = os.path.join(base_dir, '')

    all_files = glob.glob(os.path.join(base_dir, '*.jpg'))
    all_frame_ids = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in all_files)
    max_frame_id = max(all_frame_ids)

    zfill_amount = len(str(max_frame_id))

    nearby_frames = []

    for i in range(keyframe_id - n, keyframe_id):
        if i > 0:  # Ensure frame number is positive
            nearby_frames.append(f"{prefix}{str(i).zfill(zfill_amount)}.jpg")

    nearby_frames.append(keyframe_path)

    for i in range(keyframe_id + 1, keyframe_id + 1 + n):
        if i <= max_frame_id:
            nearby_frames.append(f"{prefix}{str(i).zfill(zfill_amount)}.jpg")

    return nearby_frames         




