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

    def search(self, query, k):
        query = preprocess_text(query)

        vectorize = self.tfidf_transform.transform([query])
        scores = vectorize.dot(self.context_sparse_matrix.T).toarray()[0]
        idx_audio = np.argsort(scores)[::-1][:k]
        scores = scores[idx_audio]

        print(idx_audio)
        print(scores)

        image_scores = {}

        for i, audio_idx in enumerate(idx_audio):
            image_indices = self.audio_id2id[audio_idx]
            for image_idx in image_indices:
                image_scores[image_idx] = image_scores.get(image_idx, 0) + scores[i]/k
        
        top_image_indices = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        result_image_indices = [idx for idx, score in top_image_indices]
        result_scores = [score for idx, score in top_image_indices]
        keyframe_paths = [self.idx2keyframe[idx] for idx in result_image_indices]

        return result_scores, keyframe_paths, result_image_indices