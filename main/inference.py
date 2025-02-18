import argparse
import json
import os
import pdb
import shutil
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import util.yaml_utils as yaml_utils
from util.train_util import load_models


def calc_top5_acc(pred, t):
    top5_preds = pred.argsort()[:, -5:]
    return np.asarray(np.any(top5_preds.T == t, axis=0).mean(dtype="f"))


def main():
    parser = argparse.ArgumentParser(description="Openset Concept Bottleneck Model")
    parser.add_argument("--config_path", type=str, default="configs/base.yml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="./result/", help="directory to save the results to")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--experiment_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    pattern = yaml_utils.make_pattern(config)
    out_path = os.path.join(args.results_dir, pattern, f"expr{args.experiment_id}")
    os.makedirs(out_path, exist_ok=True)
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    # Model
    classifier = load_models(config.models["classifier"])
    classifier = torch.nn.DataParallel(classifier, device_ids=[0])
    classifier.to(device)

    # Dataset
    config["dataset"]["args"]["transform"] = classifier.module.img_preprocess
    test_dataset = yaml_utils.load_dataset(config["dataset"], test=True)
    test_loader = DataLoader(test_dataset, config.batchsize, shuffle=False, num_workers=16)

    # Test loop
    class_names = classifier.module.class_names
    pred_labels = []
    correct_labels = []
    feat_gaps = []
    sparsity = []
    topk_clip_scores = []
    classwise_concepts = {cls: defaultdict(int) for cls in class_names}
    classifier.eval()
    total_batch_num = len(test_loader)
    start_time = time.time()
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data, labels = data.to(device), labels.to(device)
            result_dict = classifier(data)
            pred_labels.append(result_dict["preds"].squeeze().cpu().data.numpy())
            correct_labels.append(np.array(labels.cpu().data.numpy()))
            feat_gaps.append(result_dict["feat_gap"])
            sparsity.append(result_dict["sparsity"])
            topk_clip_scores.append(result_dict["topk_clip_score"])
            if result_dict["topk_concepts"] is not None:
                for i, label in enumerate(labels.cpu().data):
                    for concept in result_dict["topk_concepts"][i]:
                        classwise_concepts[class_names[label.item()]][concept] += 1
    end_time = time.time()

    # Evaluation
    pred_labels = np.concatenate(pred_labels)
    correct_labels = np.concatenate(correct_labels)
    top1 = accuracy_score(correct_labels, pred_labels)
    # top5 = calc_top5_acc(pred_labels, correct_labels)
    precision = precision_score(correct_labels, pred_labels, average="macro")
    recall = recall_score(correct_labels, pred_labels, average="macro")
    f_score = f1_score(correct_labels, pred_labels, average="macro")
    feat_gap = np.mean(feat_gaps)
    sparsity = np.mean(sparsity)
    topk_clip_score = np.mean(topk_clip_scores)
    time_per_iter = (end_time - start_time) / total_batch_num

    # Summarize extracted concepts
    concept_appearances = {
        cls: dict(
            sorted(
                dict(classwise_concepts[cls]).items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        for cls in class_names
    }

    out_results = {
        "scores": {
            "accuracy": float(top1),
            # "top-5 accuracy": float(top5),
            "precision": float(precision),
            "recall": float(recall),
            "f-score": float(f_score),
            "feat_gap": float(feat_gap),
            "sparsity": float(sparsity),
            "topk_clip_score": float(topk_clip_score),
            "time_per_iter": float(time_per_iter),
        },
        "classwise_concept_appearance": concept_appearances,
    }

    # Report
    result_path = os.path.join(out_path, "test_result.json")
    with open(result_path, mode="w") as f:
        json.dump(out_results, f, indent=2)

    print(out_results["scores"])
    return out_results


if __name__ == "__main__":
    main()
