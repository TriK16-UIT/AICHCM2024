import faiss 
import torch
import numpy as np
import json
from utils.Translation import Translation
from utils.ObjectDetector import ObjectDetector
from utils.OCRDetector import OCRDetector
from utils.SpeechDetector import SpeechDetector
import open_clip
import re

class Th3Faiss:
    def __init__(self, bin_clip_file: str, 
                 bin_ocr_file:str, 
                 json_idx_2_keyframe_file: str, 
                 json_audio_id2id_file: str,
                 object_file: str, 
                 sparse_context_file: str,
                 sparse_context_audio_file: str, 
                 tfidf_transform_file: str,
                 tfidf_transform_audio_file):
        self.index = self.load_bin_file(bin_clip_file)
        self.idx2keyframe = self.load_json_file(json_idx_2_keyframe_file)
        self.audio_id2id = self.load_json_file(json_audio_id2id_file)

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = Translation()
        self.ObjectDetector = ObjectDetector(object_file)
        self.OCRDetector = OCRDetector(bin_ocr_file, sparse_context_file, tfidf_transform_file, self.idx2keyframe, self.__device)
        self.SpeechDetector = SpeechDetector(sparse_context_audio_file, tfidf_transform_audio_file, self.idx2keyframe, self.audio_id2id)
        # self.clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=self.__device)
        # self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k')
        # self.clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', device=self.__device, pretrained='dfn5b')
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)
    
    def load_json_file(self, json_file: str):
        with open(json_file, 'r') as file:
            js = json.load(file)
        return {int(k):v for k, v in js.items()}
        

    def search_by_text(self, query, k, class_dict, index, filter_type):
        query = self.translator.translate(query)

        print(query)

        # query_features = self.clip_model.encode([query], convert_to_tensor=True)
        query = self.clip_tokenizer([query]).to(self.__device)
        query_features = self.clip_model.encode_text(query)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_features = query_features.cpu().detach().numpy().astype(np.float32)

        if index is None:
            scores, idx_image = self.index.search(query_features, k=k)
        elif filter_type == "including":
            # Search for direct image (for DEBUG only)
            id_selector = faiss.IDSelectorArray(index)
            scores, idx_image = self.index.search(query_features, k=1, params=faiss.SearchParametersIVF(sel=id_selector)) 
        elif filter_type == "excluding":
            filter_ids = [id for id in range(self.index.ntotal) if id not in index]
            id_selector = faiss.IDSelectorArray(filter_ids)
            scores, idx_image = self.index.search(query_features, k=k, params=faiss.SearchParametersIVF(sel=id_selector))
        
        if isinstance(idx_image, int): 
            idx_image = [idx_image]
            scores = [scores]


        scores = scores.flatten()
        idx_image = idx_image.flatten()
        keyframe_paths = [self.idx2keyframe[idx] for idx in idx_image]

        #Reranking with ObjectDetector
        scores, keyframe_paths, idx_image = self.ObjectDetector.reranking(scores, keyframe_paths, idx_image, class_dict)

        return scores, keyframe_paths, idx_image
    
    def search_by_ocr(self, query, k, search_method="embedding"):
        scores, keyframe_paths, idx_image = self.OCRDetector.search(query, k, search_method)
        return scores, keyframe_paths, idx_image
    
    def search_by_speech(self, query, k):
        scores, keyframe_paths, idx_image = self.SpeechDetector.search(query, k)
        return scores, keyframe_paths, idx_image



