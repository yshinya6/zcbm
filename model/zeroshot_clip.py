import os
import pdb
import sys

import clip
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from functools import partial

from model.components.concepts import RetrievalConcept
from model.components.nn_classifier import NearestNeighboursClassifier

OPENCLIP_DATASET = {"ViT-H-14": "laion2b_s32b_b79k", "ViT-bigG-14": "laion2b_s39b_b160k"}


def load_model(model_name: str):
    if model_name.startswith("Open_"):
        arch_name = model_name.split("_")[-1]
        model, _, preprocess = open_clip.create_model_and_transforms(arch_name, OPENCLIP_DATASET[arch_name])
        tokenizer = open_clip.get_tokenizer(arch_name)
    elif model_name == "siglip":
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    elif model_name == "dfn":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-H-14-378-quickgelu", pretrained="dfn5b")
        tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    else:  # Load CLIP models
        model, preprocess = clip.load(model_name)
        tokenizer = partial(clip.tokenize, truncate=True)
    return model, preprocess, tokenizer


class ZeroshotCLIP(torch.nn.Module):

    def __init__(
        self,
        *args,
        classname_file: str,
        backbone_name: str = "ViT-B/32",
        class_prompt: str = "a photo of a",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Load Backbone Vision-Language Model
        self.backbone, self.img_preprocess, self.tokenizer = load_model(backbone_name)

        # Load class name dict for each label id
        self.id_classname_dict = self._make_classname_dict(classname_file)
        self.class_names = list(self.id_classname_dict.values())

        self.class_prompt = class_prompt
        self._classname_features = None

    def _make_classname_dict(self, metadata_path: str):
        id_classname_dict = {}
        with open(metadata_path, "r") as f:
            for line in f:
                assert len(line.split("\t")) == 2, "metadata must be composed of lines of <class id>\\t<classname>"
                cls_id, cls_name = line.split("\t")
                id_classname_dict[cls_id] = cls_name.replace("\n", "").replace("_", " ")
        return id_classname_dict

    def get_text_features(self, device) -> torch.Tensor:
        if self._classname_features is not None:
            return self._classname_features

        class_texts = torch.cat([self.tokenizer(f"{self.class_prompt} {c}") for c in self.class_names]).to(device)
        text_features = self.backbone.encode_text(class_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._classname_features = text_features
        return self._classname_features

    def forward(self, image):
        # 1. Extract image features
        image_features = self.backbone.encode_image(image)
        nomalized_img_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. Predict final labels
        class_similarities = (100.0 * nomalized_img_features @ self.get_text_features(image.device).T).softmax(dim=-1)
        probs, preds = class_similarities.topk(1, dim=-1)

        feat_gap = 0
        pred_classnames = [self.id_classname_dict[str(pred.item())] for pred in preds]
        output = {
            "preds": preds,
            "probs": probs,
            "pred_classnames": pred_classnames,
            "topk_concepts": None,
            "topk_scores": None,
            "topk_clip_score": 0.0,
            "feat_gap": feat_gap,
            "sparsity": 1.0,
        }
        return output


if __name__ == "__main__":
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZeroshotCLIP(
        backbone_name="ViT-B/32",
        classname_file="data/classnames/test.txt",
    )
    bshort_hair_img = model.img_preprocess(Image.open("asset/british_short.jpg")).unsqueeze(0).to(device)
    ragdoll_img = model.img_preprocess(Image.open("asset/ragdoll.jpg")).unsqueeze(0).to(device)
    images = torch.cat([bshort_hair_img, ragdoll_img], dim=0)
    with torch.no_grad():
        output = model(images)
    print(output)
