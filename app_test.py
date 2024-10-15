import streamlit as st
from streamlit_dash import image_select
from utils.FAISS import Th3Faiss
import json
import os
from utils.utils import get_nearby_frames, extract_video_id_and_frame_idx
import csv
import io
import re
from functools import lru_cache

st.set_page_config(layout="wide")
st.title("Image Retrieval System")

# Cache the loading of FAISS and data
@st.cache_resource
def load_faiss_and_data():
    bin_file = "DataPreprocessing/faiss_clip_h14.bin"
    blip_bin_file = "DataPreprocessing/faiss_blip_vitg.bin"
    json_file = "DataPreprocessing/idx2keyframe.json"
    json_audio_file = "DataPreprocessing/audio_id2id.json"
    json_keyframe_mapper_file = "DataPreprocessing/map_keyframes.json"
    json_object_file = "DataPreprocessing/object.json"
    json_classes_file = "DataPreprocessing/object_classes.json"
    pkl_tfidf_transform_file = "DataPreprocessing/tfidf_transform_ocr.pkl"
    pkl_tfidf_transform_audio_file = "DataPreprocessing/tfidf_transform_audio.pkl"
    npz_sparse_context_matrix_ocr_file = "DataPreprocessing/sparse_context_matrix_ocr.npz"
    npz_sparse_context_matrix_audio_file = "DataPreprocessing/sparse_context_matrix_audio.npz"
    
    my_faiss = Th3Faiss(bin_file, blip_bin_file, json_file, json_audio_file, json_object_file, 
                        npz_sparse_context_matrix_ocr_file, npz_sparse_context_matrix_audio_file, 
                        pkl_tfidf_transform_file, pkl_tfidf_transform_audio_file)
    
    with open(json_keyframe_mapper_file, 'r') as file:
        keyframeMapper = json.load(file)
    with open(json_classes_file, 'r') as file:
        classesList = json.load(file)
    
    return my_faiss, keyframeMapper, classesList

my_faiss, keyframeMapper, classesList = load_faiss_and_data()

def get_images_from_query(query, k, search_type, model, class_dict_str, index, filter_type):
    print(class_dict_str)
    if search_type == "text":
        scores, keyframe_paths, idx_images = my_faiss.search_by_text(query, k, class_dict_str, index, filter_type, model)
    elif search_type == "speech":
        scores, keyframe_paths, idx_images = my_faiss.search_by_speech(query, k)
    else:
        scores, keyframe_paths, idx_images = my_faiss.search_by_ocr(query, k)

    return [(path, f"Score: {score:.4f} \n Index: {idx} \n Path: {os.path.splitext(os.path.relpath(path, start='Data/improved_keyframes'))[0]} \n Frame_idx: {extract_video_id_and_frame_idx(path, keyframeMapper)[1]}")
            for score, path, idx in zip(scores, keyframe_paths, idx_images)]

def download_as_csv(keyframe_paths):
    output = io.StringIO()
    writer = csv.writer(output)
    for path in keyframe_paths:
        video_id, frame_idx = extract_video_id_and_frame_idx(path, keyframeMapper)
        writer.writerow([video_id, frame_idx])
    return output.getvalue()

def sort_images_by_video_id(images_with_captions):
    sorted_images = {}
    for path, caption in images_with_captions:
        video_id, _ = extract_video_id_and_frame_idx(path, keyframeMapper)
        sorted_images.setdefault(video_id, []).append((path, caption))
    return sorted_images

def map_selected_indices_to_global(selected_local_indices, video_images, global_images):
    return [next(i for i, (path, _) in enumerate(global_images) if path == video_images[local_idx][0])
            for local_idx in selected_local_indices]

# Initialize session state
for key in ['images', 'expanded_images', 'selected_search_images', 'selected_expanded_images', 
            'expand_count', 'class_dict', 'accumulated_unselected_indices', 'search_type']:
    if key not in st.session_state:
        st.session_state[key] = [] if key.startswith('selected') or key in ['images', 'expanded_images', 'accumulated_unselected_indices'] else 3 if key == 'expand_count' else {} if key == 'class_dict' else None

# Sidebar components
sideb = st.sidebar
warning_con = st.container()

with sideb.container(border=True):
    query = st.text_input("Input query:")
    idx = st.number_input("Index:", min_value=0, value=None)

with sideb.container(border=True):
    model = st.selectbox("Select Model", options=["clip", "blip"], index=0)

with sideb.container(border=True):
    col1, col2 = st.columns(2)
    selected_object = col1.selectbox("Select a object", classesList)
    quantity = col2.number_input("Quantity", min_value=1, value=1, step=1)
    if col1.button("Add"):
        st.session_state.class_dict[selected_object] = quantity
    if col2.button("Clear"):
        st.session_state.class_dict.clear()

st.write("Selected Objects:")
st.markdown(" ".join([f"{obj}: {count}" for obj, count in st.session_state.class_dict.items()]))

with sideb.container(border=True):
    k = st.number_input("No. of images:", min_value=1, value=10)
    col5, col6 = st.columns(2)
    search_type = col5.selectbox("Search Type", options=["text", "ocr", "speech"], index=0)
    sort_type = st.selectbox("Sort Type", options=["default", "video_id"], index=0)
    if col6.button("Search"):
        if query and k:
            st.session_state.search_type = search_type
            st.session_state.images = get_images_from_query(query, k, st.session_state.search_type, model, 
                                                            st.session_state.class_dict, idx, "including")
            st.session_state.expanded_images = []
            st.session_state.selected_search_images = []
            st.session_state.selected_expanded_images = []
            st.session_state.accumulated_unselected_indices = []
        else:
            warning_con.warning("Please enter query and No. of images first")

# Tabs for search results and expanded results
tabs = st.tabs(["Search Results", "Expanded Images"])

# Display images and allow selection
with tabs[0]:
    if st.session_state.images:
        st.subheader("Search Results")
        if sort_type == "default":
            st.session_state.selected_search_images = image_select(
                "Select one or more images:", 
                [path for path, _ in st.session_state.images], 
                captions=[caption for _, caption in st.session_state.images],
                return_value="index",
                use_container_width=False
            )
        else:
            sorted_images = sort_images_by_video_id(st.session_state.images)
            temp_selected_indices = []
            for video_id, images in sorted_images.items():
                st.subheader(f"Video ID: {video_id}")
                selected_local_indices = image_select(
                    f"Select one or more images for video {video_id}:", 
                    [path for path, _ in images], 
                    captions=[caption for _, caption in images],
                    return_value="index",
                    use_container_width=False
                )
                temp_selected_indices.extend(map_selected_indices_to_global(selected_local_indices, images, st.session_state.images))
            st.session_state.selected_search_images = temp_selected_indices

        if len(st.session_state.images) != 1 and sideb.button("Search Again with Feedback"):
            if st.session_state.search_type == "text":
                unselected_indices = [int(re.search(r'Index: (\d+)', caption).group(1))
                                      for i, (_, caption) in enumerate(st.session_state.images)
                                      if i not in st.session_state.selected_search_images]
                st.session_state.accumulated_unselected_indices.extend(unselected_indices)
                st.session_state.images = get_images_from_query(query, k, st.session_state.search_type, model,
                                                                st.session_state.class_dict,
                                                                st.session_state.accumulated_unselected_indices, "excluding")
                st.session_state.selected_search_images = []
                st.rerun()
            else:
                warning_con.warning("Search again is currently available for Text method only")

# Expand functionality
with sideb.container(border=True):
    col7, col8 = st.columns(2)
    st.session_state.expand_count = col7.number_input("No. of nearby frames", min_value=1, value=st.session_state.expand_count)
    if col8.button("Expand"):
        if len(st.session_state.selected_search_images) == 1:
            selected_path = st.session_state.images[st.session_state.selected_search_images[0]][0]
            nearby_frames = get_nearby_frames(selected_path, st.session_state.expand_count)
            st.session_state.expanded_images = [(frame, f"Video ID: {video_id}\nFrame Index: {frame_idx}\nPath: {os.path.splitext(os.path.relpath(frame, start='Data/improved_keyframes'))[0]}")
                                                for frame in nearby_frames
                                                for video_id, frame_idx in [extract_video_id_and_frame_idx(frame, keyframeMapper)]]
        elif not st.session_state.selected_search_images:
            warning_con.warning("Please select an image to expand.")
        else:
            warning_con.warning("Please select only one image to expand.")

with tabs[1]:
    if st.session_state.expanded_images:
        st.subheader("Expanded Images")
        st.session_state.selected_expanded_images = image_select(
            "Select one or more expanded images:", 
            [path for path, _ in st.session_state.expanded_images], 
            captions=[caption for _, caption in st.session_state.expanded_images],
            return_value="index",
            use_container_width=False
        )

# Download results functionality
if st.session_state.images or st.session_state.expanded_images:
    selected_images = ([st.session_state.images[i] for i in st.session_state.selected_search_images if i < len(st.session_state.images)] +
                       [st.session_state.expanded_images[i] for i in st.session_state.selected_expanded_images if i < len(st.session_state.expanded_images)])
    
    csv_data = download_as_csv([image[0] for image in (selected_images if selected_images else st.session_state.images)])
    download_label = "Download Selected Results as CSV" if selected_images else "Download All Results as CSV"
    
    sideb.download_button(
        label=download_label,
        data=csv_data,
        file_name="search_results.csv",
        mime="text/csv"
    )