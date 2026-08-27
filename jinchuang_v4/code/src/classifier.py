import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import numpy as np


class ImageClassifier:
    """基于 SigLIP2 的零样本影像分类器（面签照片/身份证/权证/合同/其他）"""

    def __init__(self, model, processor, categories: list[dict], device="cpu"):
        """
        Args:
            model: SigLIP2/AutoModel 实例（复用外部已加载的模型）
            processor: AutoProcessor 实例
            categories: config.yaml classifier.categories 的列表
            device: 设备
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.categories = categories

        # 为每个类别生成文本特征（取多条 prompt 的平均）
        self._build_text_features()

    def _build_text_features(self):
        """将所有类别的文本 prompt 编码为特征向量，每条 prompt 独立"""
        all_prompts = []
        self._cat_index = []  # (cat_idx, prompt_idx) 映射

        for ci, cat in enumerate(self.categories):
            for pi, prompt in enumerate(cat["prompts"]):
                all_prompts.append(prompt)
                self._cat_index.append((ci, pi))

        if not all_prompts:
            raise ValueError("至少需要定义一个类别")

        # 使用 processor 的 tokenizer 方式
        # truncation=True：配置里带默认 max_length 且仅 padding 时不生效，会触发
        # transformers 的 UserWarning；显式声明截断策略后 max_length 才被尊重。
        texts = self.processor(
            text=all_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**texts)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features, dim=-1)

        # 按类别聚合：取该类下所有 prompt 特征的最大值（更鲁棒）
        num_cats = len(self.categories)
        cat_features = []
        for ci in range(num_cats):
            mask = [idx for idx, (c, _) in enumerate(self._cat_index) if c == ci]
            if mask:
                feats = text_features[mask]  # [num_prompts, D]
                # 取平均
                cat_features.append(feats.mean(dim=0, keepdim=True))
            else:
                cat_features.append(torch.zeros(1, text_features.size(-1), device=self.device))

        self.text_features = torch.cat(cat_features, dim=0)  # [num_cats, D]

    def classify(self, image_tensor: torch.Tensor):
        """对单张图片进行零样本分类

        Args:
            image_tensor: [1, 3, 224, 224] 归一化后的 tensor

        Returns:
            cat_id: 类别 ID (str)
            cat_name: 类别名称 (str)
            scores: 各类别相似度分数 dict {cat_name: score}
        """
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=image_tensor)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = F.normalize(image_features, dim=-1)

        # 与各类别文本特征计算余弦相似度
        similarity = (image_features @ self.text_features.T)  # [1, num_cats]
        scores = similarity[0].cpu().numpy()

        best_idx = int(np.argmax(scores))
        best_cat = self.categories[best_idx]

        score_dict = {
            cat["name"]: float(scores[i])
            for i, cat in enumerate(self.categories)
        }

        return best_cat["id"], best_cat["name"], score_dict

    def is_sign_photo(self, image_tensor: torch.Tensor):
        """判断是否为面签照片

        Args:
            image_tensor: [1, 3, 224, 224] 归一化后的 tensor

        Returns:
            is_sign: bool (预测类别为"面签照片")
            confidence: float (面签类别的相似度分数)
        """
        cat_id, cat_name, scores = self.classify(image_tensor)
        sign_score = scores.get("面签照片", 0.0)
        return cat_name == "面签照片", sign_score


# 简单测试
if __name__ == "__main__":
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    import numpy as np

    device = "cpu"
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model.eval()

    categories = [
        {"id": "sign_photo", "name": "面签照片", "prompts": ["a person signing a document", "a bank customer signing"]},
        {"id": "id_card", "name": "身份证", "prompts": ["a photo of an ID card"]},
        {"id": "contract", "name": "合同", "prompts": ["a scanned contract document"]},
    ]

    clf = ImageClassifier(model, processor, categories, device=device)

    # 创建一张测试图
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    cat_id, cat_name, scores = clf.classify(img_tensor)
    print(f"分类结果: {cat_name} ({cat_id})")
    print(f"各类别得分: {scores}")
