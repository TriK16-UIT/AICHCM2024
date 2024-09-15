import json

class ObjectDetector:
    def __init__(self, object_dict_path):
        with open(object_dict_path, 'r') as f:
            self.object_dict = json.load(f)

    def reranking(self, scores, keyframe_paths, idx_image, class_dict):
        if not class_dict:
            return scores, keyframe_paths, idx_image

        point = 1 / sum(class_dict.values())

        top_k = {idx: {"score": score * 0.8, "path": path} for idx, score, path in zip(idx_image, scores, keyframe_paths)}

        for key, value in top_k.items():
            minus = 0
            for obj, count in class_dict.items():
                if str(key) in self.object_dict and obj in self.object_dict[str(key)]:
                    minus += abs(count - self.object_dict[str(key)][obj]['count']) * point
                else:
                    minus += count * point
            value['score'] += (1 - minus) * 0.2

        sorted_results = sorted(top_k.items(), key=lambda x: x[1]['score'], reverse=True)

        reranked_idx_image = [item[0] for item in sorted_results]
        reranked_scores = [item[1]['score'] for item in sorted_results]
        reranked_keyframe_paths = [item[1]['path'] for item in sorted_results]

        return reranked_scores, reranked_keyframe_paths, reranked_idx_image
