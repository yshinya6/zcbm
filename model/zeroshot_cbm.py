import os
import pdb
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import clip
import jax.numpy as jnp
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.utils._testing import ignore_warnings
from skscope import HTPSolver, ScopeSolver
from skscope.utilities import LinearSIC
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from model.components.concepts import RetrievalConcept
from model.components.nn_classifier import NearestNeighboursClassifier

warnings.filterwarnings("ignore")
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


class ZCBM(torch.nn.Module):

    def __init__(
        self,
        *args,
        classname_file: str,
        backbone_name: str = "ViT-B/32",
        concept_index: str = "index/wordnet_raw_concept_ViT-B32_index.bin",
        metadata_path: str = "concepts/wordnet_raw_concept.txt",
        class_prompt: str = "a photo of a",
        num_concept_candidates: int = 512,
        num_present_concepts: int = 32,
        concept_sparsity: int = 256,
        selection_strategy: str = "lasso",
        prediction_strategy: str = "linear_regression",
        alpha: float = 0.01,
        clip_score_weight: float = 2.5,
        bilevel_concept_prediction: bool = False,
        use_faiss_gpu: bool = False,
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
        self.selection_strategy = selection_strategy
        self.prediction_strategy = prediction_strategy
        self.alpha = alpha
        self.num_present_concepts = num_present_concepts
        self.clip_score_weight = clip_score_weight

        self.concept_selector = self.set_regressor(selection_strategy)
        self.bilevel_concept_prediction = bilevel_concept_prediction
        self.score_predictor = self.set_regressor(prediction_strategy)

        self.concept_sparsity = concept_sparsity

        self.conceptbase = RetrievalConcept(
            concept_index=concept_index,
            metadata_path=metadata_path,
            num_samples=num_concept_candidates,
            use_faiss_gpu=use_faiss_gpu,
        )

    def _make_classname_dict(self, metadata_path: str):
        id_classname_dict = {}
        with open(metadata_path, "r") as f:
            for line in f:
                assert len(line.split("\t")) == 2, "metadata must be composed of lines of <class id>\\t<classname>"
                cls_id, cls_name = line.split("\t")
                id_classname_dict[cls_id] = cls_name.replace("\n", "").replace("_", " ")
        return id_classname_dict

    def similarity_prediction(self, retrieved_dicts, image_features, indices=None) -> torch.Tensor:
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_similarities = np.array(
                [ret["similarities"][index] for ret, index in zip(retrieved_dicts, indices)]
            )
        else:
            concept_similarities = torch.tensor(np.array([ret["similarities"] for ret in retrieved_dicts]))
        concept_score = concept_similarities.softmax(dim=-1)
        return concept_score

    @ignore_warnings(category=ConvergenceWarning)
    def linear_regression_prediction(self, retrieved_dicts, image_features, indices=None) -> torch.Tensor:
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_array = np.array([ret["embeddings"][index] for ret, index in zip(retrieved_dicts, indices)])
        else:
            concept_array = np.array([ret["embeddings"] for ret in retrieved_dicts])
        image_array = image_features.cpu().float().numpy()
        assert len(concept_array) == len(image_array)

        def fit_linear_regression(i):
            lasso = LinearRegression()
            concept_feats = concept_array[i].transpose(1, 0)
            image_feat = image_array[i]
            lasso.fit(concept_feats, image_feat)
            return lasso.coef_

        with ThreadPoolExecutor() as executor:
            concept_score = list(executor.map(fit_linear_regression, range(len(concept_array))))

        concept_score = torch.from_numpy(np.vstack(concept_score))
        return concept_score

    @ignore_warnings(category=ConvergenceWarning)
    def lasso_prediction(self, retrieved_dicts, image_features, indices=None) -> torch.Tensor:
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_array = np.array([ret["embeddings"][index] for ret, index in zip(retrieved_dicts, indices)])
        else:
            concept_array = np.array([ret["embeddings"] for ret in retrieved_dicts])
        image_array = image_features.cpu().float().numpy()
        assert len(concept_array) == len(image_array)

        def fit_lasso(i):
            lasso = Lasso(alpha=self.alpha, fit_intercept=False)
            concept_feats = concept_array[i].transpose(1, 0)
            image_feat = image_array[i]
            lasso.fit(concept_feats, image_feat)
            return lasso.coef_

        with ThreadPoolExecutor(max_workers=4) as executor:
            concept_score = list(executor.map(fit_lasso, range(len(concept_array))))

        concept_score = torch.from_numpy(np.vstack(concept_score))
        return concept_score

    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=LinAlgWarning)
    def ridge_prediction(self, retrieved_dicts, image_features, indices=None):
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_array = np.array([ret["embeddings"][index] for ret, index in zip(retrieved_dicts, indices)])
        else:
            concept_array = np.array([ret["embeddings"] for ret in retrieved_dicts])
        image_array = image_features.cpu().float().numpy()
        assert len(concept_array) == len(image_array)

        def fit_ridge(i):
            ridge = Ridge(alpha=self.alpha, fit_intercept=False)
            concept_feats = concept_array[i].transpose(1, 0)
            image_feat = image_array[i]
            ridge.fit(concept_feats, image_feat)
            return ridge.coef_

        with ThreadPoolExecutor() as executor:
            concept_score = list(executor.map(fit_ridge, range(len(concept_array))))

        concept_score = torch.from_numpy(np.vstack(concept_score))
        return concept_score

    @ignore_warnings(category=ConvergenceWarning)
    def elasticnet_prediction(self, retrieved_dicts, image_features, indices=None) -> torch.Tensor:
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_array = np.array([ret["embeddings"][index] for ret, index in zip(retrieved_dicts, indices)])
        else:
            concept_array = np.array([ret["embeddings"] for ret in retrieved_dicts])
        image_array = image_features.cpu().float().numpy()
        assert len(concept_array) == len(image_array)

        def fit_elasticnet(i):
            elasticnet = ElasticNet(self.alpha, fit_intercept=False)
            concept_feats = concept_array[i].transpose(1, 0)
            image_feat = image_array[i]
            elasticnet.fit(concept_feats, image_feat)
            return elasticnet.coef_

        with ThreadPoolExecutor() as executor:
            concept_score = list(executor.map(fit_elasticnet, range(len(concept_array))))

        concept_score = torch.from_numpy(np.vstack(concept_score))
        return concept_score

    def htp_prediction(self, retrieved_dicts, image_features, indices=None) -> torch.Tensor:
        if indices is not None:
            assert len(indices) == len(retrieved_dicts)
            concept_array = np.array([ret["embeddings"][index] for ret, index in zip(retrieved_dicts, indices)])
        else:
            concept_array = np.array([ret["embeddings"] for ret in retrieved_dicts])
        image_array = image_features.cpu().float().numpy()
        assert len(concept_array) == len(image_array)

        def fit_htp(i):
            concept_feats = concept_array[i].transpose(1, 0)
            solver = HTPSolver(
                concept_feats.shape[1],
                sparsity=self.concept_sparsity,
                ic_method=LinearSIC,
                sample_size=concept_feats.shape[0],
            )
            image_feat = image_array[i]

            def ols_loss(params):
                loss = jnp.mean((image_feat - concept_feats @ params) ** 2)
                return loss

            coef = solver.solve(ols_loss, jit=True)
            return coef

        with ThreadPoolExecutor() as executor:
            concept_score = list(executor.map(fit_htp, range(len(concept_array))))

        concept_score = torch.from_numpy(np.vstack(concept_score)).float()
        return concept_score

    def set_regressor(self, strategy):
        match strategy:
            case "linear_regression":
                return self.linear_regression_prediction
            case "lasso":
                return self.lasso_prediction
            case "ridge":
                return self.ridge_prediction
            case "elasticnet":
                return self.elasticnet_prediction
            case "htp":
                return self.htp_prediction
            case _:
                return self.similarity_prediction

    def get_text_features(self, device) -> torch.Tensor:
        if self._classname_features is not None:
            return self._classname_features

        class_texts = torch.cat([self.tokenizer(f"{self.class_prompt} {c}") for c in self.class_names]).to(device)
        text_features = self.backbone.encode_text(class_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._classname_features = text_features.cpu().float()
        return self._classname_features

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
        class_similarities = (100.0 * predicted_features @ self.get_text_features(image.device).T).softmax(dim=-1)
        probs, preds = class_similarities.topk(1, dim=-1)

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