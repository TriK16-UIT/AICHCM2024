import faiss
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
    def __init__(self, ocr_bin_file, sparse_context_file, tfidf_transform_file, idx2keyframe, device):
        self.ocr_index = faiss.read_index(ocr_bin_file)
        self.context_sparse_matrix = scipy.sparse.load_npz(sparse_context_file)
        self.idx2keyframe = idx2keyframe
        with open(tfidf_transform_file, 'rb') as f:
            self.tfidf_transform = pickle.load(f)
        self.ocr_model = SentenceTransformer('dangvantuan/vietnamese-embedding', device=device)
    def search(self, query, k, search_method):
        query = preprocess_text(query)

        print(query)

        if search_method == "embedding":
            tokenizer = tokenize(query)
            truncated_tokenizer = [tokenizer[:512]]
            ocr_features = self.ocr_model.encode(truncated_tokenizer, convert_to_tensor=True)
            ocr_features /= ocr_features.norm(dim=-1, keepdim=True)
            ocr_features = ocr_features.cpu().detach().numpy().astype(np.float32)

            scores, idx_image = self.ocr_index.search(ocr_features, k=k)
            scores = scores.flatten()
            idx_image = idx_image.flatten()
            keyframe_paths = [self.idx2keyframe[idx] for idx in idx_image]
        elif search_method == "tfidf":
            vectorize = self.tfidf_transform.transform([query])
            scores = vectorize.dot(self.context_sparse_matrix.T).toarray()[0]
            idx_image = np.argsort(scores)[::-1][:k]
            scores = scores[idx_image]
            keyframe_paths = [self.idx2keyframe[idx] for idx in idx_image]

        return scores, keyframe_paths, idx_image