# Dataset Extraction
## Data directory
Firstly, prepare data as:
```bash
|- Data
  | - videos
    | - L01_V001.mp4
    | - L01_V002.mp4
    | - ...
```
## Usage
- Scene extraction: [extract_scenes.ipynb](DataPreprocessing/extract_scenes.ipynb)
- Keyframe extraction: [extract_keyframes.ipynb](DataPreprocessing/extract_keyframes.ipynb)
- CLIP feature extraction: [extract_clip_features.ipynb](DataPreprocessing/extract_clip_features.ipynb)
- BLIP feature extraction: [extract_blip_features.ipynb](DataPreprocessing/extract_blip_features.ipynb)
- OCR extraction: [extract_ocr.py](DataPreprocessing/PaddleOCR/extract_ocr.py)
- OD extraction: [saving_class_dict.py](DataPreprocessing/Yolov10-AIChallenge/saving_class_dict.py)
- Audio extraction: [extract_audio.py](DataPreprocessing/PaddleOCR/extract_audio.py)
- Finally, run all create_*.py before using the app.
