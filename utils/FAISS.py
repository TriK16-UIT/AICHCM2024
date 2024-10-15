import faiss
import numpy as np
import json
import torch
from utils.Translation import Translation
from utils.ObjectDetector import ObjectDetector
from utils.OCRDetector import OCRDetector
from utils.SpeechDetector import SpeechDetector
from utils.models import BaseModel

class Th3Faiss:
    def __init__(self, bin_clip_file: str,
                 bin_blip_file: str, 
                 json_idx_2_keyframe_file: str, 
                 json_audio_id2id_file: str,
                 object_file: str, 
                 sparse_context_file: str,
                 sparse_context_audio_file: str, 
                 tfidf_transform_file: str,
                 tfidf_transform_audio_file):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.index = self.load_bin_file(bin_clip_file)
        self.blip_index = self.load_bin_file(bin_blip_file)

        self.idx2keyframe = self.load_json_file(json_idx_2_keyframe_file)
        self.audio_id2id = self.load_json_file(json_audio_id2id_file)

        self.translator = Translation()
        self.ObjectDetector = ObjectDetector(object_file)
        self.OCRDetector = OCRDetector(sparse_context_file, tfidf_transform_file, self.idx2keyframe)
        self.SpeechDetector = SpeechDetector(sparse_context_audio_file, tfidf_transform_audio_file, self.idx2keyframe, self.audio_id2id)
    
        self.clip_model = BaseModel(model_name="clip", device=self.device)
        self.blip_model = BaseModel(model_name="blip", device=self.device)

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)
    
    def load_json_file(self, json_file: str):
        with open(json_file, 'r') as file:
            js = json.load(file)
        return {int(k):v for k, v in js.items()}
        

    def search_by_text(self, query, k, class_dict, index, filter_type, model):
        query = self.translator.translate(query)

        print(query)

        if model == "clip":   
           query_features = self.clip_model.encode_query(query)
        elif model == "blip":
            query_features = self.blip_model.encode_query(query)
            
        query_features = query_features.cpu().detach().numpy().astype(np.float32)

        if model == "clip":
            chosen_model = self.index
        elif model == "blip":
            chosen_model = self.blip_index

        if index is None:
            scores, idx_image = chosen_model.search(query_features, k=k)
        elif filter_type == "including":
            # Search for direct image (for DEBUG only)
            id_selector = faiss.IDSelectorArray(index)
            scores, idx_image = chosen_model.search(query_features, k=1, params=faiss.SearchParametersIVF(sel=id_selector)) 
        elif filter_type == "excluding":
            filter_ids = [id for id in range(self.index.ntotal) if id not in index]
            id_selector = faiss.IDSelectorArray(filter_ids)
            scores, idx_image = chosen_model.search(query_features, k=k, params=faiss.SearchParametersIVF(sel=id_selector))
        
        if isinstance(idx_image, int): 
            idx_image = [idx_image]
            scores = [scores]

        scores = scores.flatten()
        idx_image = idx_image.flatten()
        keyframe_paths = [self.idx2keyframe[idx] for idx in idx_image]

        #Reranking with ObjectDetector
        scores, keyframe_paths, idx_image = self.ObjectDetector.reranking(scores, keyframe_paths, idx_image, class_dict)

        return scores, keyframe_paths, idx_image
    
    def search_by_ocr(self, query, k):
        scores, keyframe_paths, idx_image = self.OCRDetector.search(query, k)
        return scores, keyframe_paths, idx_image
    
    def search_by_speech(self, query, k):
        scores, keyframe_paths, idx_image = self.SpeechDetector.search(query, k)
        return scores, keyframe_paths, idx_image



