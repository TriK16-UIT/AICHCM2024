from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils.FAISS import Th3Faiss
import json
import os
from utils.utils import get_nearby_frames, extract_video_id_and_frame_idx
import csv
import io

app = Flask(__name__)
CORS(app)

# Initialize Faiss and load data
def load_faiss_and_data():
    bin_file = "DataPreprocessing/faiss_clip_h14.bin"
    blip_bin_file = "DataPreprocessing/faiss_blip_vitg.bin"
    ocr_bin_file = "DataPreprocessing/faiss_ocr.bin"
    json_file = "DataPreprocessing/idx2keyframe.json"
    json_audio_file = "DataPreprocessing/audio_id2id.json"
    json_keyframe_mapper_file = "DataPreprocessing/map_keyframes.json"
    json_object_file = "DataPreprocessing/object.json"
    json_classes_file = "DataPreprocessing/object_classes.json"
    pkl_tfidf_transform_file = "DataPreprocessing/tfidf_transform_ocr.pkl"
    pkl_tfidf_transform_audio_file = "DataPreprocessing/tfidf_transform_audio.pkl"
    npz_sparse_context_matrix_ocr_file = "DataPreprocessing/sparse_context_matrix_ocr.npz"
    npz_sparse_context_matrix_audio_file = "DataPreprocessing/sparse_context_matrix_audio.npz"
    
    my_faiss = Th3Faiss(bin_file, blip_bin_file, ocr_bin_file, json_file, json_audio_file, json_object_file, npz_sparse_context_matrix_ocr_file, npz_sparse_context_matrix_audio_file, pkl_tfidf_transform_file, pkl_tfidf_transform_audio_file)
    
    with open(json_keyframe_mapper_file, 'r') as file:
        keyframeMapper = json.load(file)
    with open(json_classes_file, 'r') as file:
        classesList = json.load(file)
    
    return my_faiss, keyframeMapper, classesList

my_faiss, keyframeMapper, classesList = load_faiss_and_data()

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']
    k = data['k']
    search_type = data['search_type']
    class_dict = data.get('class_dict', {})
    index = data.get('index')
    filter_type = data.get('filter_type', 'including')
    model = data.get('model', 'clip')

    if search_type == "text":
        scores, keyframe_paths, idx_images = my_faiss.search_by_text(query, k, class_dict, index, filter_type, model)
    elif search_type == "speech":
        scores, keyframe_paths, idx_images = my_faiss.search_by_speech(query, k)
    else:
        search_type = search_type.replace("ocr_", "")
        scores, keyframe_paths, idx_images = my_faiss.search_by_ocr(query, k, search_type)

    images_with_captions = []
    for score, path, idx in zip(scores, keyframe_paths, idx_images):
        formatted_path = os.path.splitext(os.path.relpath(path, start="..Data/improved_keyframes"))[0]
        caption = f"Score: {score:.4f} \n Index: {idx} \n Path: {formatted_path} \n Frame_idx: {extract_video_id_and_frame_idx(path, keyframeMapper)[1]}"
        images_with_captions.append({"path": path, "caption": caption})
    
    return jsonify(images_with_captions)

@app.route('/api/expand', methods=['POST'])
def expand():
    data = request.json
    selected_path = data['selected_path']
    expand_count = data['expand_count']

    nearby_frames = get_nearby_frames(selected_path, expand_count)
    expanded_images = []
    for frame in nearby_frames:
        video_id, frame_idx = extract_video_id_and_frame_idx(frame, keyframeMapper)
        formatted_path = os.path.splitext(os.path.relpath(frame, start="Data/improved_keyframes"))[0]
        caption = f"Video ID: {video_id}\nFrame Index: {frame_idx}\nPath: {formatted_path}"
        expanded_images.append({"path": frame, "caption": caption})

    return jsonify(expanded_images)

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify(classesList)

@app.route('/api/download', methods=['POST'])
def download_csv():
    data = request.json
    images = data['images']

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['video_id', 'frame_idx'])

    for image in images:
        video_id, frame_idx = extract_video_id_and_frame_idx(image['path'], keyframeMapper)
        writer.writerow([video_id, frame_idx])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename='search_results.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)