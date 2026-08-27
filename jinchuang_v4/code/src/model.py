import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"


class SigLIP2FeatureExtractor:
    def __init__(self, device="cpu", model_name: str = DEFAULT_MODEL_NAME):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto").to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def extract(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=images)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)
        return features


if __name__ == "__main__":
    extractor = SigLIP2FeatureExtractor()
    dummy = torch.randn(2, 3, 224, 224)
    out = extractor.extract(dummy)
    print(f"Feature extraction OK: {out.shape}")
