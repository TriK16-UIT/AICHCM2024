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

# Initialize Faiss and load data
@st.cache_resource
def load_faiss_and_data():
    bin_file = "DataPreprocessing/faiss_clip_l14.bin"
    ocr_bin_file = "DataPreprocessing/faiss_ocr.bin"
    json_file = "DataPreprocessing/idx2keyframe.json"
    json_keyframe_mapper_file = "DataPreprocessing/map_keyframes.json"
    json_object_file = "DataPreprocessing/object.json"
    json_classes_file = "DataPreprocessing/object_classes.json"
    pkl_tfidf_transform_file = "DataPreprocessing/tfidf_transform_ocr.pkl"
    npz_sparse_context_matrix_ocr_file = "DataPreprocessing/sparse_context_matrix_ocr.npz"
    
    my_faiss = Th3Faiss(bin_file, ocr_bin_file, json_file, json_object_file, npz_sparse_context_matrix_ocr_file, pkl_tfidf_transform_file)
    
    with open(json_keyframe_mapper_file, 'r') as file:
        keyframeMapper = json.load(file)
    with open(json_classes_file, 'r') as file:
        classesList = json.load(file)
    
    return my_faiss, keyframeMapper, classesList

my_faiss, keyframeMapper, classesList = load_faiss_and_data()

# Function to get images from query
def get_images_from_query(query, k, search_type="text", class_dict={}):
    if search_type == "text":
        scores, keyframe_paths, idx_images = my_faiss.search_by_text(query, k, class_dict)
    else:
        search_type = search_type.replace("ocr_", "")
        scores, keyframe_paths, idx_images = my_faiss.search_by_ocr(query, k, search_type)

    images_with_captions = []
    for score, path, idx in zip(scores, keyframe_paths, idx_images):
        formatted_path = os.path.splitext(os.path.relpath(path, start="Data/improved_keyframes"))[0]
        caption = f"Score: {score:.4f} \n Index: {idx} \n Path: {formatted_path} \n Frame_idx: {extract_video_id_and_frame_idx(path, keyframeMapper)[1]}"
        images_with_captions.append((path, caption))
    
    return scores, keyframe_paths, idx_images, images_with_captions

# Function to save results as CSV
def download_as_csv(keyframe_paths):
    output = io.StringIO()
    writer = csv.writer(output)
    for path in keyframe_paths:
        video_id, frame_idx = extract_video_id_and_frame_idx(path, keyframeMapper)
        writer.writerow([video_id, frame_idx])
    return output.getvalue()

# Streamlit app
st.title("Image Retrieval System")

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
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'class_dict' not in st.session_state:
    st.session_state.class_dict = {}

# Search interface
query = st.text_input("Input query:")

st.subheader("Select Objects and Quantities")
col1, col2 = st.columns(2)

with col1:
    selected_object = st.selectbox("Select an object", classesList)

with col2:
    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

if st.button("Add Object"):
    st.session_state.class_dict[selected_object] = quantity

# Display selected objects
st.write("Selected Objects:")
for obj, count in st.session_state.class_dict.items():
    st.write(f"{obj}: {count}")

if st.button("Clear Selection"):
    st.session_state.class_dict = {}

k = st.number_input("No. of images:", min_value=1, value=10)

search_type = st.selectbox("Search Type", options=["text", "ocr_embedding", "ocr_tfidf"], index=0)

if st.button("Search"):
    if query and k:
        print(st.session_state.class_dict)
        scores, keyframe_paths, idx_images, images_with_captions = get_images_from_query(query, k, search_type, st.session_state.class_dict)
        st.session_state.search_results = (scores, keyframe_paths, idx_images)
        st.session_state.images = images_with_captions
        st.session_state.expanded_images = []

# Display images and allow selection
if st.session_state.images:
    st.subheader("Search Results")
    st.session_state.selected_search_images = image_select(
        "Select one or more images:", 
        [path for path, _ in st.session_state.images], 
        captions=[caption for _, caption in st.session_state.images],
        return_value="index",
        use_container_width=True
    )

# Expand functionality
st.subheader("Expand Images")
st.session_state.expand_count = st.number_input("Enter the number of nearby frames to expand:", min_value=1, value=st.session_state.expand_count)

if st.button("Expand"):
    if st.session_state.selected_search_images and len(st.session_state.selected_search_images) == 1:
        selected_path = st.session_state.images[st.session_state.selected_search_images[0]][0]
        nearby_frames = get_nearby_frames(selected_path, st.session_state.expand_count)
        st.write("Expanded images:")
        
        st.session_state.expanded_images = []
        cols = st.columns(3)
        for i, frame in enumerate(nearby_frames):                
            # Extract and display relevant information
            video_id, frame_idx = extract_video_id_and_frame_idx(frame, keyframeMapper)
            formatted_path = os.path.splitext(os.path.relpath(frame, start="Data/improved_keyframes"))[0]
            caption = f"Video ID: {video_id}\nFrame Index: {frame_idx}\nPath: {formatted_path}"
            st.session_state.expanded_images.append((frame, caption))
    elif not st.session_state.selected_search_images:
        st.warning("Please select an image to expand.")
    else:
        st.warning("Please select only one image to expand.")

if st.session_state.expanded_images:
    st.subheader("Expanded Images")
    st.session_state.selected_expanded_images = image_select(
        "Select one or more expanded images:", 
        [path for path, _ in st.session_state.expanded_images], 
        captions=[caption for _, caption in st.session_state.expanded_images],
        return_value="index",
        use_container_width=True
    )

# Download results functionality
if st.session_state.images or st.session_state.expanded_images:
    selected_search_images = [st.session_state.images[i] for i in st.session_state.selected_search_images] if st.session_state.selected_search_images else []
    selected_expanded_images = [st.session_state.expanded_images[i] for i in st.session_state.selected_expanded_images] if st.session_state.selected_expanded_images else []
    
    all_selected_images = selected_search_images + selected_expanded_images
    
    if all_selected_images:
        csv_data = download_as_csv([image[0] for image in all_selected_images])
        download_label = "Download Selected Results as CSV"
    else:
        # all_images = st.session_state.images + st.session_state.expanded_images
        csv_data = download_as_csv([image[0] for image in st.session_state.images])
        download_label = "Download All Results as CSV"
    
    st.download_button(
        label=download_label,
        data=csv_data,
        file_name="search_results.csv",
        mime="text/csv"
    )