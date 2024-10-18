import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_dash import image_select
from utils.FAISS import Th3Faiss
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utils.utils import get_nearby_frames, extract_video_id_and_info
from config import USERNAME, PASSWORD, VIDEOS_PATH, METADATA_PATH
from st_aggrid import AgGrid, ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_paste_button import paste_image_button as pbutton
import csv
import io
import re
import pandas as pd
import requests

st.set_page_config(layout="wide")
st.title("Image Retrieval System")

sideb = st.sidebar
warning_con = st.container()

# Cache the loading of FAISS and data
@st.cache_resource
def load_faiss_and_data():
    bin_file = "DataPreprocessing/faiss_clip_h14.bin"
    blip_bin_file = "DataPreprocessing/faiss_blip_vitg.bin"
    json_file = "DataPreprocessing/idx2keyframe.json"
    json_audio_file = "DataPreprocessing/audio_id2id.json"
    json_keyframe_mapper_file = "DataPreprocessing/map_keyframes_final.json"
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
    print(index)
    if search_type == "text":
        scores, keyframe_paths, idx_images = my_faiss.search_by_text(query, k, class_dict_str, index, filter_type, model)
    elif search_type == "speech":
        scores, keyframe_paths, idx_images = my_faiss.search_by_speech(query, k, class_dict_str, index, filter_type)
    elif search_type == "ocr": 
        scores, keyframe_paths, idx_images = my_faiss.search_by_ocr(query, k, class_dict_str, index, filter_type)
    elif search_type == "image":
        scores, keyframe_paths, idx_images = my_faiss.search_by_image(query, k, class_dict_str, index, filter_type, model)
 
    return [(path, f"Score: {score:.4f} \n Index: {idx} \n Path: {os.path.splitext(os.path.relpath(path, start='Data/improved_keyframes'))[0]} \n Frame_idx: {extract_video_id_and_info(path, keyframeMapper)[1]}")
            for score, path, idx in zip(scores, keyframe_paths, idx_images)]

def download_as_csv(keyframe_paths):
    output = io.StringIO()
    writer = csv.writer(output)
    for path in keyframe_paths:
        video_id, frame_idx, _ = extract_video_id_and_info(path, keyframeMapper)
        writer.writerow([video_id, frame_idx])
    return output.getvalue()

def get_youtube_link(video_id):
    with open(video_id, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('watch_url')

def sort_images_by_video_id(images_with_captions):
    sorted_images = {}
    for path, caption in images_with_captions:
        video_id, _, _ = extract_video_id_and_info(path, keyframeMapper)
        sorted_images.setdefault(video_id, []).append((path, caption))
    return sorted_images

def map_selected_indices_to_global(selected_local_indices, video_images, global_images):
    return [next(i for i, (path, _) in enumerate(global_images) if path == video_images[local_idx][0])
            for local_idx in selected_local_indices]

def login(username, password):
    try:
        login_url = "https://eventretrieval.one/api/v2/login"
        login_payload = {
            "username": username,
            "password": password
        }
        headers = {'Content-Type': 'application/json'}
        
        login_response = requests.post(login_url, headers=headers, data=json.dumps(login_payload))
        login_response.raise_for_status()
        
        session_data = login_response.json()
        session_id = session_data.get('sessionId')

        return session_id

    except requests.exceptions.HTTPError as http_err:
        warning_con.warning(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        warning_con.warning(f"Other error occurred: {err}")
        return None

def fetch_question(session_id):
    try:
        evaluation_url = f"https://eventretrieval.one/api/v2/client/evaluation/list?session={session_id}"
        evaluation_response = requests.get(evaluation_url)
        evaluation_response.raise_for_status() 
        
        evaluation_data = evaluation_response.json()
        task_to_id_map = {task["name"]: item["id"] for item in evaluation_data for task in item["taskTemplates"]}
    except requests.exceptions.HTTPError as http_err:
        warning_con.warning(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        warning_con.warning(f"Other error occurred: {err}")
        return None
    return task_to_id_map
    
def submission(df, submission_type, session_id, evaluation_id):
    answer_sets = []
    if submission_type == "QA":
        for index, row in df.iterrows():
            answer_text = f"{row['QA']}-{row['Video ID']}-{row['pts_time']}"
            answer_sets.append({"text": answer_text})
        
        submission_body = {
            "answerSets": [
                {
                    "answers": answer_sets
                }
            ]
        }
    elif submission_type == "KIS":
        for index, row in df.iterrows():
            answer_sets.append({
                "mediaItemName": row['Video ID'],
                "start": row['pts_time'],
                "end": row['end_time']
            })

        submission_body = {
            "answerSets": [
                {
                    "answers": answer_sets
                }
            ]
        }

    url = f"https://eventretrieval.one/api/v2/submit/{evaluation_id}?session={session_id}"
    headers = {
        'Content-Type': 'application/json'
    }

    st.write(submission_body)
    response = requests.post(url, headers=headers, data=json.dumps(submission_body))

    return response.status_code, response.text


# Initialize session state
for key in ['images', 'expanded_images', 'selected_search_images', 'selected_expanded_images', 
            'expand_count', 'class_dict', 'accumulated_unselected_indices', 'search_type']:
    if key not in st.session_state:
        st.session_state[key] = [] if key.startswith('selected') or key in ['images', 'expanded_images', 'accumulated_unselected_indices'] else 3 if key == 'expand_count' else {} if key == 'class_dict' else None
if 'session_id' not in st.session_state:
    session_id = login(USERNAME, PASSWORD)
    if session_id:
        st.session_state.session_id = session_id
    else:
        st.session_state.session_id = None
else:
    warning_con.success(f"Login successful!")

with sideb.container(border=True):
    query = st.text_input("Input query:")
    idx = st.number_input("Index:", min_value=0, value=None)
    image = pbutton(label="Upload image from Clipboard", errors="raise").image_data
    if image:
        st.success("Sucessfully Uploaded!")

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
    search_type = col5.selectbox("Search Type", options=["text", "ocr", "speech", "image"], index=0)
    sort_type = st.selectbox("Sort Type", options=["default", "video_id"], index=0)
    if col6.button("Search"):
        if k:
            if search_type in ["text", "ocr", "speech"] and query:
                st.session_state.search_type = search_type
                st.session_state.images = get_images_from_query(query, k, st.session_state.search_type, model, 
                                                                st.session_state.class_dict, idx, "including")
            elif search_type == "image" and image:
                st.session_state.search_type = search_type
                st.session_state.images = get_images_from_query(image, k, st.session_state.search_type, model, 
                                                                st.session_state.class_dict, idx, "including")
            st.session_state.expanded_images = []
            st.session_state.selected_search_images = []
            st.session_state.selected_expanded_images = []
            st.session_state.accumulated_unselected_indices = []
        
        else:
            warning_con.warning("Please enter query or select an image, and specify the number of images (k)")

# Tabs for search results and expanded results
tabs = st.tabs(["Search Results", "Expanded Images", "Submissions", "Authorization"])

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
            unselected_indices = [int(re.search(r'Index: (\d+)', caption).group(1))
                                    for i, (_, caption) in enumerate(st.session_state.images)
                                    if i not in st.session_state.selected_search_images]
            st.session_state.accumulated_unselected_indices.extend(unselected_indices)
            if st.session_state.search_type in ["text", "ocr", "speech"]:
                st.session_state.images = get_images_from_query(query, k, st.session_state.search_type, model,
                                                                st.session_state.class_dict,
                                                                st.session_state.accumulated_unselected_indices, "excluding")
            elif st.session_state.search_type == "image":
                st.session_state.images = get_images_from_query(image, k, st.session_state.search_type, model,
                                                                st.session_state.class_dict,
                                                                st.session_state.accumulated_unselected_indices, "excluding")
            st.session_state.selected_search_images = []
            st.rerun()

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
                                                for video_id, frame_idx, _ in [extract_video_id_and_info(frame, keyframeMapper)]]
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
with tabs[2]:
    col1, col2 = st.columns([1, 1])
    sel_row = None
    with col1:
        st.subheader("Submissions")
        task_to_id_map = fetch_question(st.session_state.session_id)
        if not task_to_id_map:
            warning_con.warning("Please head to Authorization tab for new sessionID")
        else:
            task_name = st.selectbox("Select Task", options=list(task_to_id_map.keys()))
            task_id = task_to_id_map[task_name]
            if "qa" in task_name:
                submission_type = "QA"
            elif "kis" in task_name:
                submission_type = "KIS"
            else:
                submission_type = "KIS"
            if st.session_state.images or st.session_state.expanded_images:
                selected_images = ([st.session_state.images[i] for i in st.session_state.selected_search_images if i < len(st.session_state.images)] +
                            [st.session_state.expanded_images[i] for i in st.session_state.selected_expanded_images if i < len(st.session_state.expanded_images)])
                if not selected_images:
                    selected_images = st.session_state.images
                new_data = []
                for path, _ in selected_images:
                    video_id, frame_idx, pts_time = extract_video_id_and_info(path, keyframeMapper)
                    new_data.append({
                        'Video ID': str(video_id),
                        'Frame Index': str(frame_idx),
                        'pts_time': int(pts_time*1000),
                        'Path': os.path.relpath(path, "Data/improved_keyframes")
                    })
                df = pd.DataFrame(new_data).drop_duplicates()
                if submission_type == "QA":
                    df.insert(3, "QA", "")
                elif submission_type == "KIS":
                    df.insert(3, "end_time", df["pts_time"])
                # edited_df = st.data_editor(df, num_rows="dynamic", hide_index=False)

                gd = GridOptionsBuilder.from_dataframe(df)
                gd.configure_pagination(enabled=True)
                gd.configure_default_column(editable=True, groupable=True, resizable=False)
                gd.configure_selection(selection_mode='single')
                gridoptions = gd.build()
                edited_df = AgGrid(df, gridOptions=gridoptions, allow_unsafe_jscode=True, fit_columns_on_grid_load=True, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)

                sel_row = edited_df["selected_rows"]
                if st.button("Submit"):
                    st.write(sel_row)
                    if isinstance(sel_row, pd.DataFrame):
                        st.write(submission(sel_row, submission_type, st.session_state.session_id, task_id))
                    else:
                        st.warning("Please select a row for submission")
            else:
                st.write("No submissions yet. Please select images")
    with col2:
        if isinstance(sel_row, pd.DataFrame):
            st.subheader("Preview")
            st.write("Preview cutscene before submitting")
            video = sel_row['Video ID'].iloc[0]
            start_time = sel_row['pts_time'].iloc[0] / 1000
            if os.path.exists(VIDEOS_PATH):
                video = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")
                st.video(data=video, start_time=start_time)
            else:
                st.warning("Local video not available. Streaming from Youtube...")
                video = os.path.join(METADATA_PATH, f"{video_id}.json")
                video = get_youtube_link(video)
                st.video(data=video, start_time=start_time)
       
with tabs[3]:
    st.subheader("Authorization")
    st.write("In case there is error with sessionID. Login again to get new ones")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("Login"):
            st.markdown("#### Enter your credentials")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password") 
            submit = st.form_submit_button("Login")
    if submit:
        session_id = login(username, password)
        if session_id:
            st.session_state.session_id = session_id
            st.rerun()

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