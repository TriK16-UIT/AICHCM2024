import re
import pickle
import scipy
import numpy as np
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

def preprocess_text(text: str):
    text = text.lower()
    reg_pattern = r'[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸÝửữựỳỵỷỹý\s]'
    output = re.sub(reg_pattern, '', text)
    output = output.strip()
    output = " ".join(output.split())
    return output

class OCRDetector:
    def __init__(self, sparse_context_file, tfidf_transform_file, idx2keyframe):
        self.context_sparse_matrix = scipy.sparse.load_npz(sparse_context_file)
        self.idx2keyframe = idx2keyframe
        with open(tfidf_transform_file, 'rb') as f:
            self.tfidf_transform = pickle.load(f)
    def search(self, query, k):
        query = preprocess_text(query)

        print(query)

        vectorize = self.tfidf_transform.transform([query])
        scores = vectorize.dot(self.context_sparse_matrix.T).toarray()[0]
        idx_image = np.argsort(scores)[::-1][:k]
        scores = scores[idx_image]
        keyframe_paths = [self.idx2keyframe[idx] for idx in idx_image]

        return scores, keyframe_paths, idx_image