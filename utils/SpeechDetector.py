import faiss
import re
import pickle
import scipy
import numpy as np

def preprocess_text(text: str):
    text = text.lower()
    reg_pattern = r'[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸÝửữựỳỵỷỹý\s]'
    output = re.sub(reg_pattern, '', text)
    output = output.strip()
    output = " ".join(output.split())
    return output

class SpeechDetector:
    def __init__(self, sparse_context_file, tfidf_transform_file, idx2keyframe, audio_id2id):
        self.context_sparse_matrix = scipy.sparse.load_npz(sparse_context_file)
        self.idx2keyframe = idx2keyframe
        self.audio_id2id = audio_id2id
        with open(tfidf_transform_file, 'rb') as f:
            self.tfidf_transform = pickle.load(f)

    def map_image_to_audio(self, image_indices):
        audio_indices = set()
        for img_idx in image_indices:
            for audio_idx, img_list in self.audio_id2id.items():
                if img_idx in img_list:
                    audio_indices.add(audio_idx)
        return list(audio_indices)

    def search(self, query, k, index, filter_type):
        query = preprocess_text(query)

        vectorize = self.tfidf_transform.transform([query])
        scores = vectorize.dot(self.context_sparse_matrix.T).toarray()[0]

        result_image_indices = []
        result_scores = []
        
        if filter_type == "including" and index is not None:
            for audio_idx, image_indices in self.audio_id2id.items():
                if index in image_indices:
                    result_image_indices = [index]
                    result_scores = [scores[audio_idx]]
                    keyframe_paths = [self.idx2keyframe[index]]
                    return result_scores, keyframe_paths, result_image_indices
        else:
            idx_audio = np.argsort(scores)[::-1]

            result_image_indices = []
            result_scores = []

            for audio_idx in idx_audio:
                if len(result_image_indices) >= k:
                    break

                image_indices = self.audio_id2id[audio_idx]
                
                if filter_type == "excluding" and index is not None:
                    image_indices = [img_idx for img_idx in image_indices if img_idx not in index]

                for img_idx in image_indices:
                    if len(result_image_indices) >= k:
                        break
                    result_image_indices.append(img_idx)
                    result_scores.append(scores[audio_idx]) 

        keyframe_paths = [self.idx2keyframe[idx] for idx in result_image_indices]

        return result_scores, keyframe_paths, result_image_indices