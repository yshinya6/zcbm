import argparse
import os
import pdb
import pickle
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk import RegexpParser, pos_tag, word_tokenize
from tqdm import tqdm

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def save_concept_list(outpath, concept_list):
    with open(outpath, "w") as f:
        f.write(concept_list[0])
        for concept in concept_list[1:]:
            f.write("\n" + concept)


def extract_noun_phrase(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    grammer = "NP: {<DT>?<JJ>*<NN.*>+}"
    cp = RegexpParser(grammer)
    tree = cp.parse(tagged_tokens)

    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            np = " ".join(word for word, tag in subtree.leaves() if tag != "DT")
            noun_phrases.append(np)

    return noun_phrases


def extract_flickr30k_noun_phrase_concept(args):
    df = pd.read_csv(args.metadata_path, delimiter=",", header=None)
    concept_set = set()
    text_list = df[1]
    for text in tqdm(text_list):
        if not isinstance(text, str):
            continue
        noun_phrases = extract_noun_phrase(text)
        concept_set |= set(noun_phrases)
    concept_list = list(concept_set)
    concept_list.sort()
    outpath = Path(args.output_dir) / f"{args.dataset}.txt"
    save_concept_list(outpath, concept_list)
    print(f"Completed to extrat concepts. Concept vocabulary size: {len(concept_list)}")


def extract_cc3m_noun_phrase_concept(args):
    df = pd.read_csv(args.metadata_path, delimiter="\t", header=None)
    concept_set = set()
    text_list = df[0]
    for text in tqdm(text_list):
        if not isinstance(text, str):
            continue
        noun_phrases = extract_noun_phrase(text)
        concept_set |= set(noun_phrases)
    concept_list = list(concept_set)
    concept_list.sort()
    outpath = Path(args.output_dir) / f"{args.dataset}.txt"
    save_concept_list(outpath, concept_list)
    print(f"Completed to extrat concepts. Concept vocabulary size: {len(concept_list)}")


def extract_cc12m_noun_phrase_concept(args):
    df = pd.read_csv(args.metadata_path, delimiter="\t", header=None)
    concept_set = set()
    text_list = df[1]
    for text in tqdm(text_list):
        if not isinstance(text, str):
            continue
        noun_phrases = extract_noun_phrase(text)
        concept_set |= set(noun_phrases)
    concept_list = list(concept_set)
    concept_list.sort()
    outpath = Path(args.output_dir) / f"{args.dataset}.txt"
    save_concept_list(outpath, concept_list)
    print(f"Completed to extrat concepts. Concept vocabulary size: {len(concept_list)}")


def extract_yfcc15m_noun_phrase_concept(args):
    df = pd.read_json(args.metadata_path, lines=True)
    concept_set = set()
    text_list = df["caption"]
    for text in tqdm(text_list):
        if not isinstance(text, str):
            continue
        noun_phrases = extract_noun_phrase(text)
        concept_set |= set(noun_phrases)
    concept_list = list(concept_set)
    concept_list.sort()
    outpath = Path(args.output_dir) / f"{args.dataset}.txt"
    save_concept_list(outpath, concept_list)
    print(f"Completed to extrat concepts. Concept vocabulary size: {len(concept_list)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cc3m_multilabel", help="dataset as concept source")
    parser.add_argument("--output_dir", type=str, default="./base_concepts/", help="directory to save the results to")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="./metadata/CC3M/Image_Labels_Subset_Train_GCC-Labels-training.tsv",
        help="dataset as concept source",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "cc3m":
        extract_cc3m_noun_phrase_concept(args)
    elif args.dataset == "flickr30k":
        extract_flickr30k_noun_phrase_concept(args)
    elif args.dataset == "cc12m":
        extract_cc12m_noun_phrase_concept(args)
    elif args.dataset == "yfcc15m":
        extract_yfcc15m_noun_phrase_concept(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
