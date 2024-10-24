# Dataset Extraction
## Data directory
Firstly, prepare data as:
```bash
|- Data
  | - videos
    | - L01_V001.mp4
    | - L01_V002.mp4
    | - ...
  | - media-info
    | - L01_V001.json
    | - L01_V002.json
    | - ...
```
## Usage
- Scene extraction: [extract_scenes.ipynb](extract_scenes.ipynb)
- Keyframe extraction: [extract_keyframes.ipynb](extract_keyframes.ipynb)
- CLIP feature extraction: [extract_clip_features.ipynb](extract_clip_features.ipynb)
- BLIP feature extraction: [extract_blip_features.ipynb](extract_blip_features.ipynb)
- OCR extraction: [extract_ocr.py](PaddleOCR/extract_ocr.py)
- OD extraction: [saving_class_dict.py](Yolov10-AIChallenge/saving_class_dict.py)
- Audio extraction: [extract_audio.py](PaddleOCR/extract_audio.py)
- Finally, run all create_*.py before using the app.

**NOTE:**
For OCR and OD, you need to clone the official github (PaddleOCR & Yolov10) to run the preprocessing files (conflicts with main env).
