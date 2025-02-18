import argparse
import os
import pdb
import pickle
import sys
from functools import partial
from pathlib import Path

import clip
import faiss
import nltk
import numpy as np
import open_clip
import pandas as pd
import torch
from nltk import RegexpParser, pos_tag, word_tokenize
from nltk.corpus import wordnet
from tqdm import tqdm

from util import conceptset_utils

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

"""
CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
MAX_LEN: max number of characters in a concept

PRINT_PROB: what percentage of filtered concepts will be printed
"""

CLASS_SIM_CUTOFF = 0.9
OTHER_SIM_CUTOFF = 0.95
MAX_LEN = 20
PRINT_PROB = 0.01

OPENCLIP_DATASET = {"ViT-H-14": "laion2b_s32b_b79k", "ViT-bigG-14": "laion2b_s39b_b160k"}


def load_model(model_name: str, device: str):
    if model_name.startswith("Open_"):
        arch_name = model_name.split("_")[-1]
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch_name, OPENCLIP_DATASET[arch_name], device=device
        )
        tokenizer = open_clip.get_tokenizer(arch_name)
    elif model_name == "siglip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:timm/ViT-SO400M-14-SigLIP-384", device=device
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    elif model_name == "dfn":
        model, preprocess = open_clip.create_model_from_pretrained(
            "ViT-H-14-378-quickgelu", pretrained="dfn5b", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    else:  # Load CLIP models
        model, preprocess = clip.load(model_name, device=device)
        tokenizer = partial(clip.tokenize, truncate=True)
    return model, tokenizer


def generate_text_embeddings(concepts, batchsize, model, device, tokenizer):
    text_embeddings = []

    for i in tqdm(range(0, len(concepts), batchsize)):
        batch_concepts = concepts[i : i + batchsize]
        text_tokens = tokenizer(batch_concepts).to(device)
        with torch.no_grad():
            batch_embs = model.encode_text(text_tokens)
            batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
            text_embeddings.append(batch_embs.cpu().numpy())

    text_embeddings = np.vstack(text_embeddings)
    return text_embeddings


def create_faiss_index(concepts, model, batchsize, tokenizer, use_gpu=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#### Extracting concept text features...")
    concept_embeddings = generate_text_embeddings(concepts, batchsize, model, device, tokenizer)
    emb_dim = concept_embeddings.shape[1]
    print("#### Creating FAISS index...")
    index = faiss.IndexFlatIP(emb_dim)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(concept_embeddings)
    return index


def load_classname_list(inpath):
    classnames = []
    with open(inpath, "r") as f:
        for line in f:
            cls_name = line.split("\t")[-1].lower()
            classnames.append(cls_name.replace("\n", "").replace("_", " "))
    return classnames


def load_concept_list(inpath):
    with open(inpath, "r") as f:
        concepts = [line.strip().lower() for line in f]
    return concepts


def save_concept_list(outpath, concept_list):
    with open(outpath, "w") as f:
        f.write(concept_list[0])
        for concept in concept_list[1:]:
            f.write("\n" + concept)


def construct_concept(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("### Loading base concepts...")
    # Load base concept
    concept_set = set()
    for base_concept in args.base_concepts:
        concept_set |= set(load_concept_list(base_concept))
    # Load user-defined concept
    if args.user_concept is not None:
        concept_set |= set(load_concept_list(args.user_concept))
    # Load class name list
    if args.target_label is not None:
        classnames = load_classname_list(args.target_label)
        # Subtract class names from concepts
        concept_set -= set(classnames)
        dataset_name = Path(args.target_label).stem
    else:
        dataset_name = "nolabel"
    concepts = list(concept_set)
    print(f"### Base concept size: {len(concept_set)}")

    print("### Filtering out too long concepts...")
    concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)
    print(f"### Current concept size: {len(concepts)}")

    model, tokenizer = load_model(args.model, device=device)
    if args.filtering_similar or args.target_label is not None:
        print("### Creating concept index for filtering...")
        if args.precomputed_index is None:
            index = create_faiss_index(concepts, model, args.batchsize, tokenizer, use_gpu=args.use_faiss_gpu)
        else:
            index = faiss.read_index(args.precomputed_index, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

    if args.filtering_similar:
        print("### Filtering out too similar concepts to each other concepts...")
        similar_concepts = conceptset_utils.filter_too_similar_faiss(
            concepts,
            index,
            model,
            OTHER_SIM_CUTOFF,
            PRINT_PROB,
            batchsize=args.batchsize,
            sampling_prob=args.sampling_prob,
        )
        print(f"### Current concept size: {len(concepts) - len(similar_concepts)}")
    else:
        similar_concepts = set()

    if args.target_label is not None:
        print("### Filtering out too similar concepts to class names...")
        similar_concepts_cls = conceptset_utils.filter_too_similar_to_cls_faiss(
            concepts, classnames, index, model, CLASS_SIM_CUTOFF, PRINT_PROB
        )
    else:
        similar_concepts_cls = set()

    concept_set = set(concepts)
    concept_set -= similar_concepts
    concept_set -= similar_concepts_cls
    print(f"### Current concept size: {len(concept_set)}")

    base_concept_names = [Path(p).stem for p in args.base_concepts]
    outpath = Path(args.output_dir) / f"{'_'.join(base_concept_names)}_{dataset_name}.txt"
    save_concept_list(outpath, list(concept_set))
    index = create_faiss_index(concepts=list(concept_set), model=model, batchsize=args.batchsize, tokenizer=tokenizer)
    index_outpath = (
        Path("index") / f"merged_{'_'.join(base_concept_names)}_{dataset_name}_{args.model.replace('/','')}_index.bin"
    )
    faiss.write_index(index, str(index_outpath))
    print(f"Completed to extrat concepts. Concept vocabulary size: {len(concept_set)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="vision language backbone model name",
        choices=["ViT-B/32", "ViT-L/14", "Open_ViT-H-14", "Open_ViT-bigG-14", "siglip", "dfn"],
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="list of target class labels",
    )
    parser.add_argument(
        "--base_concepts",
        nargs="+",
        type=str,
        default=["concepts/cc3m_noun_phrase_raw_concept.txt"],
        help="list of base concepts",
    )
    parser.add_argument(
        "--user_concept",
        type=str,
        default=None,
        help="list of user concepts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./concept_bank/",
        help="directory to save the results to",
    )
    parser.add_argument(
        "--precomputed_index",
        type=str,
        default=None,
        help="directory to save the results to",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=512,
        help="batchsize for extracting features",
    )
    parser.add_argument(
        "--filtering_similar",
        action="store_true",
    )
    parser.add_argument(
        "--use_faiss_gpu",
        action="store_true",
    )
    parser.add_argument(
        "--sampling_prob",
        type=float,
        default=1.0,
        help="sampling probability for compute similarity",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    construct_concept(args)


if __name__ == "__main__":
    main()
