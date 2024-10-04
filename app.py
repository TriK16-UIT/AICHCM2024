import streamlit as st
# from streamlit_image_select import image_select
from streamlit_dash import image_select
from utils.FAISS import Th3Faiss
import json
from PIL import Image
import os
from utils.utils import get_nearby_frames, extract_video_id_and_frame_idx
import csv
import io
import re

# with open("static/index.html", "r") as file:
#     custom_setting = file.read()

# Streamlit app
# st.markdown(custom_setting, unsafe_allow_html=True)
st.set_page_config(layout="wide")
st.title("Image Retrieval System")
warning_con = st.container()
sideb = st.sidebar

# Initialize Faiss and load data
@st.cache_resource
def load_faiss_and_data():
    bin_file = "DataPreprocessing/faiss_clip_h14.bin"
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
    
    my_faiss = Th3Faiss(bin_file, ocr_bin_file, json_file, json_audio_file, json_object_file, npz_sparse_context_matrix_ocr_file, npz_sparse_context_matrix_audio_file, pkl_tfidf_transform_file, pkl_tfidf_transform_audio_file)
    
    with open(json_keyframe_mapper_file, 'r') as file:
        keyframeMapper = json.load(file)
    with open(json_classes_file, 'r') as file:
        classesList = json.load(file)
    
    return my_faiss, keyframeMapper, classesList

my_faiss, keyframeMapper, classesList = load_faiss_and_data()

# Function to get images from query
def get_images_from_query(query, k, search_type="text", class_dict={}, index=None, filter_type="including"):
    print(index)
    if search_type == "text":
        scores, keyframe_paths, idx_images = my_faiss.search_by_text(query, k, class_dict, index, filter_type)
    elif search_type == "speech":
        scores, keyframe_paths, idx_images = my_faiss.search_by_speech(query, k)
    else:
        search_type = search_type.replace("ocr_", "")
        scores, keyframe_paths, idx_images = my_faiss.search_by_ocr(query, k, search_type)

    images_with_captions = []
    for score, path, idx in zip(scores, keyframe_paths, idx_images):
        formatted_path = os.path.splitext(os.path.relpath(path, start="Data/improved_keyframes"))[0]
        caption = f"Score: {score:.4f} \n Index: {idx} \n Path: {formatted_path} \n Frame_idx: {extract_video_id_and_frame_idx(path, keyframeMapper)[1]}"
        images_with_captions.append((path, caption))
    
    return images_with_captions

# Function to save results as CSV
def download_as_csv(keyframe_paths):
    output = io.StringIO()
    writer = csv.writer(output)
    for path in keyframe_paths:
        video_id, frame_idx = extract_video_id_and_frame_idx(path, keyframeMapper)
        writer.writerow([video_id, frame_idx])
    return output.getvalue()

# Group by video_id
def sort_images_by_video_id(images_with_captions):
    sorted_images = {}
    for path, caption in images_with_captions:
        video_id, _ = extract_video_id_and_frame_idx(path, keyframeMapper)
        if video_id not in sorted_images:
            sorted_images[video_id] = []
        sorted_images[video_id].append((path, caption))
    return sorted_images

# Helper function to map indices back to st.session_state.images
def map_selected_indices_to_global(selected_local_indices, video_images, global_images):
    mapped_indices = []
    for local_idx in selected_local_indices:
        selected_image_path = video_images[local_idx][0]
        global_idx = next(i for i, (path, _) in enumerate(global_images) if path == selected_image_path)
        mapped_indices.append(global_idx)
    return mapped_indices

# Initialize session state
if 'images' not in st.session_state:
    st.session_state.images = []
if 'expanded_images' not in st.session_state:
    st.session_state.expanded_images = []
if 'selected_search_images' not in st.session_state:
    st.session_state.selected_search_images = []
if 'selected_expanded_images' not in st.session_state:
    st.session_state.selected_expanded_images = []
if 'expand_count' not in st.session_state:
    st.session_state.expand_count = 3
if 'class_dict' not in st.session_state:
    st.session_state.class_dict = {}
if 'accumulated_unselected_indices' not in st.session_state:
    st.session_state.accumulated_unselected_indices = []
if 'search_type' not in st.session_state:
    st.session_state.search_type = None

# Search interface
con1 = sideb.container(border=True)
query = con1.text_input("Input query:")
idx = con1.number_input("Index:", min_value=0, value=None)

# Seject Object for reranking
con2 = sideb.container(border=True)
col1, col2 = con2.columns(2)
selected_object = col1.selectbox("Select a object", classesList)
quantity = col2.number_input("Quantity", min_value=1, value=1, step=1)
if col1.button("Add"):
    st.session_state.class_dict[selected_object] = quantity
if col2.button("Clear"):
    st.session_state.class_dict = {}

# Display selected objects
st.write("Selected Objects:")
inline_text = " ".join([f"{obj}: {count}" for obj, count in st.session_state.class_dict.items()])
st.markdown(f"<p style='display:inline'>{inline_text}</p>", unsafe_allow_html=True)

# Search method and No. of images
con3 = sideb.container(border=True)
k = con3.number_input("No. of images:", min_value=1, value=10)
col5, col6 = con3.columns(2, vertical_alignment="bottom")
search_type = col5.selectbox("Search Type", options=["text", "ocr_embedding", "ocr_tfidf", "speech"], index=0)
sort_type = con3.selectbox("Sort Type", options=["default", "video_id"], index=0)
col6.write("# ")
if col6.button("Search"):
    if query and k:
        print(st.session_state.class_dict)
        st.session_state.search_type = search_type
        images_with_captions = get_images_from_query(query, k, st.session_state.search_type, st.session_state.class_dict, idx, filter_type="including")
        st.session_state.images = images_with_captions
        st.session_state.expanded_images = []
        st.session_state.selected_search_images = []
        st.session_state.selected_expanded_images = []
        st.session_state.accumulated_unselected_indices = []
    else:
        warning_con.warning("Please enter query and No. of images first")

# Initialize tabs for search results and expanded results
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
        if sort_type == "video_id":
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
                
                mapped_indices = map_selected_indices_to_global(selected_local_indices, images, st.session_state.images)
                temp_selected_indices.extend(mapped_indices)
            
            st.session_state.selected_search_images = temp_selected_indices

        if len(st.session_state.images) != 1:
            if sideb.button("Search Again with Feedback"):
                if st.session_state.search_type == "text":
                    def extract_index(caption):
                        match = re.search(r'Index: (\d+)', caption)
                        return int(match.group(1)) if match else None
                    unselected_images = [st.session_state.images[i] for i in range(len(st.session_state.images)) if i not in st.session_state.selected_search_images]
                    unselected_indices = [extract_index(caption) for _, caption in unselected_images]

                    st.session_state.accumulated_unselected_indices.extend(unselected_indices)

                    images_with_captions = get_images_from_query(query, k, st.session_state.search_type, st.session_state.class_dict, st.session_state.accumulated_unselected_indices, filter_type="excluding")
                    st.session_state.images = images_with_captions
                    st.session_state.selected_search_images = []
                    st.rerun()
                else:
                    warning_con.warning("Search again is currently available for Text method only")

# Expand functionality
con4 = sideb.container(border=True)
col7, col8 = con4.columns(2, vertical_alignment="bottom")
st.session_state.expand_count = col7.number_input("No. of nearby frames", min_value=1, value=st.session_state.expand_count)
if col8.button("Expand"):
    if st.session_state.selected_search_images and len(st.session_state.selected_search_images) == 1:
        selected_path = st.session_state.images[st.session_state.selected_search_images[0]][0]
        nearby_frames = get_nearby_frames(selected_path, st.session_state.expand_count)
        
        st.session_state.expanded_images = []
        cols = st.columns(3)
        for i, frame in enumerate(nearby_frames):                
            # Extract and display relevant information
            video_id, frame_idx = extract_video_id_and_frame_idx(frame, keyframeMapper)
            formatted_path = os.path.splitext(os.path.relpath(frame, start="Data/improved_keyframes"))[0]
            caption = f"Video ID: {video_id}\nFrame Index: {frame_idx}\nPath: {formatted_path}"
            st.session_state.expanded_images.append((frame, caption))
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
    selected_search_images = [st.session_state.images[i] for i in st.session_state.selected_search_images if i < len(st.session_state.images)]
    selected_expanded_images = [st.session_state.expanded_images[i] for i in st.session_state.selected_expanded_images if i < len(st.session_state.expanded_images)]
    
    all_selected_images = selected_search_images + selected_expanded_images
    
    if all_selected_images:
        csv_data = download_as_csv([image[0] for image in all_selected_images])
        download_label = "Download Selected Results as CSV"
    else:
        # all_images = st.session_state.images + st.session_state.expanded_images
        csv_data = download_as_csv([image[0] for image in st.session_state.images])
        download_label = "Download All Results as CSV"
    
    sideb.download_button(
        label=download_label,
        data=csv_data,
        file_name="search_results.csv",
        mime="text/csv"
    )