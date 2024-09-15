import json
from utils.FAISS import Th3Faiss
from utils.utils import handle_multiple_inputs, handle_single_input

bin_file = "DataPreprocessing/faiss_clip.bin"
json_file = "DataPreprocessing/idx2keyframe.json"
json_keyframe_mapper_file = "DataPreprocessing/map_keyframes.json"

my_faiss = Th3Faiss(bin_file, json_file)

with open(json_keyframe_mapper_file, 'r') as file:
    keyframeMapper = json.load(file)

input_dir = "input"
output_dir = "submission"

# k = 1

# handle_multiple_inputs(input_dir, output_dir, my_faiss, keyframeMapper, k)

query = "the person take the dog for a walk"
handle_single_input(query, "query-6-qa", output_dir, my_faiss, keyframeMapper, 100)


