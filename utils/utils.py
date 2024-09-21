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

def extract_video_id_and_frame_idx(keyframe_path, keyframeMapper):
    parts = keyframe_path.split(os.sep)
    video_id = parts[-2]
    keyframe_id = os.path.splitext(parts[-1])[0]

    frame_idx = keyframeMapper.get(video_id, {}).get(str(int(keyframe_id)), 'N/A')

    return video_id, frame_idx

def handle_multiple_inputs(input_dir, output_dir, faiss, keyframeMapper, k):
    os.makedirs(output_dir, exist_ok=True)

    for input_file in os.listdir(input_dir):
        if input_file.endswith('.txt'):
            input_path = os.path.join(input_dir, input_file)
            with open(input_path, 'r', encoding='utf-8') as file:
                query = file.read().strip()

            scores, keyframe_paths, idx_images = faiss.search_by_text(query, k)

            save_results(scores, keyframe_paths, idx_images, os.path.splitext(input_file)[0], "output")
            
            lst_video_id, lst_frame_idx = zip(*[extract_video_id_and_frame_idx(path, keyframeMapper) for path in keyframe_paths])
            
            output_file = os.path.splitext(input_file)[0] + '.csv'
            output_path = os.path.join(output_dir, output_file)
            save_to_csv(output_path, lst_video_id, lst_frame_idx)

def handle_single_input(query, output_name, output_dir, faiss, keyframeMapper, k):
    os.makedirs(output_dir, exist_ok=True)
    scores, keyframe_paths, idx_images = faiss.search_by_text(query, k)

    save_results(scores, keyframe_paths, idx_images, output_name, "output")
    
    lst_video_id, lst_frame_idx = zip(*[extract_video_id_and_frame_idx(path, keyframeMapper) for path in keyframe_paths])
    
    output_path = os.path.join(output_dir, output_name + '.csv')
    save_to_csv(output_path, lst_video_id, lst_frame_idx)

def save_results(scores, keyframe_paths, idx_images, output_filename, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(len(scores) * 5, 10))  # Set the figure size
    for i, (score, path, idx_image) in enumerate(zip(scores, keyframe_paths, idx_images)):
        img = mpimg.imread(path)
        plt.subplot(1, len(scores), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Score: {score}\nPath: {path}\nIndex: {idx_image}", fontsize=10, wrap=True, ha='center')
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def get_nearby_frames(keyframe_path, n):
    ## This function will look for n nearby frames before and after the chosen frame    
    base_dir = os.path.dirname(keyframe_path)
    keyframe_id = int(os.path.splitext(os.path.basename(keyframe_path))[0])
    prefix = os.path.join(base_dir, '')

    all_files = glob.glob(os.path.join(base_dir, '*.jpg'))
    all_frame_ids = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in all_files)
    max_frame_id = max(all_frame_ids)

    nearby_frames = []

    for i in range(keyframe_id - n, keyframe_id):
        if i > 0:  # Ensure frame number is positive
            nearby_frames.append(f"{prefix}{str(i).zfill(3)}.jpg")

    nearby_frames.append(keyframe_path)

    for i in range(keyframe_id + 1, keyframe_id + 1 + n):
        if i <= max_frame_id:
            nearby_frames.append(f"{prefix}{str(i).zfill(3)}.jpg")

    return nearby_frames         




