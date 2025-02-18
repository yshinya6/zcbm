import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.zeroshot_cbm import ZCBM


class LpCLIP(nn.Module):
    def __init__(
        self,
        backbone_name="ViT-B/32",
        num_classes=1000,
    ):
        super().__init__()
        self.backbone, self.img_preprocess = clip.load(backbone_name)
        self.embed_dim = self.backbone.text_projection.shape[1]
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, self.num_classes)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        self.convert_models_to_fp32()

    def convert_models_to_fp32(self):
        for p in self.parameters():
            p.data = p.data.float()

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor):
        return self.backbone.encode_image(x)

    @torch.no_grad()
    def encode_text(self, t: torch.Tensor):
        return self.backbone.encode_text(t)

    @torch.no_grad()
    def infer(self, feat: torch.Tensor):
        pred = self.fc(feat)
        return pred

    def forward(self, x: torch.Tensor, test=False):
        with torch.no_grad():
            feat_v = self.backbone.encode_image(x)
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)
            feat_v = torch.flatten(feat_v, 1)

        pred = self.fc(feat_v)
        return pred


class LpZCBM(ZCBM):
    def __init__(
        self,
        *args,
        pretrained_path: str,
        backbone_name: str = "ViT-B/32",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backbone = LpCLIP(backbone_name=backbone_name, num_classes=num_classes)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        self.num_classes = num_classes

    def forward(self, image):
        # 1. Extract image features
        image_features = self.backbone.encode_image(image)
        nomalized_img_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. Retrieve visual concepts from image features
        retrieved = self.conceptbase(images_z=image_features)
        concept_features = torch.tensor(np.array([ret["embeddings"] for ret in retrieved]))
        concept_names = np.vstack([ret["concepts"] for ret in retrieved])

        # 3. Select concepts according to strategy
        concept_mask = self.concept_selector(retrieved, nomalized_img_features)

        # 4. Compute wegihted average concept features
        predicted_features = concept_features.mul(concept_mask.unsqueeze(dim=-1)).sum(dim=1)

        # 5. Predict final labels
        prediction_probs = self.backbone.infer(predicted_features.to(image.device)).softmax(dim=-1).cpu()
        probs, preds = prediction_probs.topk(1, dim=-1)

        # 6. Compute Scores
        pred_classnames = [self.id_classname_dict[str(pred.item())] for pred in preds]
        feat_gap = (
            F.pairwise_distance(nomalized_img_features.cpu().float(), predicted_features, p=2).mean().cpu().item()
        )
        density = np.mean(
            [
                sample_concept_mask.count_nonzero() / float(len(sample_concept_mask))
                for sample_concept_mask in concept_mask
            ]
        ).item()
        _, topk_indices = torch.abs(concept_mask).topk(self.num_present_concepts, dim=-1)
        _, topk_pos_indices = concept_mask.topk(self.num_present_concepts, dim=-1)
        topk_scores = [score[indice] for score, indice in zip(concept_mask, topk_indices)]
        topk_concepts = [
            valid_concept_names[indice] for valid_concept_names, indice in zip(concept_names, topk_indices)
        ]
        topk_mean_features = torch.stack(
            [
                torch.mean(concept_feature[index], dim=0)
                for concept_feature, index in zip(concept_features, topk_pos_indices)
            ]
        )
        mean_concept_array = topk_mean_features.cpu().numpy()
        image_feat_array = nomalized_img_features.cpu().numpy()
        topk_clip_score = np.mean(
            self.clip_score_weight * np.clip(np.sum(image_feat_array * mean_concept_array, axis=1), 0, None)
        ).item()

        output = {
            "preds": preds,
            "probs": probs,
            "pred_classnames": pred_classnames,
            "topk_concepts": topk_concepts,
            "topk_scores": topk_scores,
            "topk_clip_score": topk_clip_score,
            "feat_gap": feat_gap,
            "sparsity": 1.0 - density,
        }
        return output
