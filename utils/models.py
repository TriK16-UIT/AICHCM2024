import torch
import open_clip
from lavis.models import load_model_and_preprocess
from PIL import Image

class BaseModel:
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device

        if self.model_name == 'clip':
            self.model, self.tokenizer = self._load_clip_model()
        elif self.model_name == 'blip':
            self.model, self.vis_processors, self.txt_processors = self._load_blip_model()
            self.sample_image = Image.new('RGB', (10, 10))
            self.sample_image = self.vis_processors["eval"](self.sample_image).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def _load_clip_model(self):
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-H-14-378-quickgelu', device=self.device, pretrained='dfn5b'
        )
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        return model, tokenizer

    def _load_blip_model(self):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=self.device
        )
        return model, vis_processors, txt_processors

    def encode_query(self, query):
        if self.model_name == 'clip':
            return self._encode_clip_query(query)
        elif self.model_name == 'blip':
            return self._encode_blip_query(query)

    def _encode_clip_query(self, query):
        tokens = self.tokenizer([query]).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def _encode_blip_query(self, query):
        query_input = self.txt_processors["eval"](query)
        sample = {"image": self.sample_image, "text_input": query_input}
        features = self.model.extract_features(sample, mode="text")
        return features.text_embeds_proj[:, 0, :] / features.text_embeds_proj[:, 0, :].norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = BaseModel(model_name="clip", device=device)
    clip_features = clip_model.encode_query("A query for CLIP")

    blip_model = BaseModel(model_name="blip", device=device)
    blip_features = blip_model.encode_query("A query for BLIP")

    print("CLIP Features:", clip_features)
    print("BLIP Features:", blip_features)
