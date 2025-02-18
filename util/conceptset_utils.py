import math
import pdb
import random

import clip
import nltk
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


def get_init_conceptnet(classes, limit=200, relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
    concepts = set()

    for cls in tqdm(classes):
        words = cls.replace(",", "").split(" ")
        for word in words:
            obj = requests.get("http://api.conceptnet.io/c/en/{}?limit={}".format(word, limit)).json()
            obj.keys()
            for dicti in obj["edges"]:
                rel = dicti["rel"]["label"]
                try:
                    if dicti["start"]["language"] != "en" or dicti["end"]["language"] != "en":
                        continue
                except KeyError:
                    continue

                if rel in relations:
                    if rel in ["IsA"]:
                        concepts.add(dicti["end"]["label"])
                    else:
                        concepts.add(dicti["start"]["label"])
                        concepts.add(dicti["end"]["label"])
    return concepts


def remove_too_long(concepts, max_len, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            if random.random() < print_prob:
                print(len(concept), concept)
    print(len(concepts), len(new_concepts))
    return new_concepts


def remove_proper_noun_concepts(concepts):
    new_concepts = []
    for concept in concepts:
        tokens = nltk.word_tokenize(concept)
        pos_tags = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(pos_tags, binary=True)
        delete_flag = False
        for chunk in named_entities:
            if hasattr(chunk, "label") and chunk.label() == "NE":
                delete_flag = True
        if not delete_flag:
            new_concepts.append(concept)
    return new_concepts


def filter_too_similar_to_cls(concepts, classes, sim_cutoff, device="cuda", print_prob=0):
    # first check simple text matches
    print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)

    # Remove proper noun classes only
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix + cls)
                if random.random() < print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix + cls))
            except ValueError:
                pass
        try:
            concepts.remove(cls.upper())
        except ValueError:
            pass
        try:
            concepts.remove(cls[0].upper() + cls[1:])
        except ValueError:
            pass
    print(len(concepts))

    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts)
    # weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i, j]
            if prod >= sim_cutoff and i != j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random() < print_prob:
                        print(
                            "Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(
                                classes[i], concepts[j], dot_prods[i, j], concepts[j]
                            )
                        )
                        print("".format(concepts[j]))

    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def filter_too_similar_to_cls_faiss(concepts, classes, index, model, sim_cutoff, print_prob=0, device="cuda"):
    # Remove proper noun classes only
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix + cls)
                if random.random() < print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix + cls))
            except ValueError:
                pass
        try:
            concepts.remove(cls.upper())
        except ValueError:
            pass
        try:
            concepts.remove(cls[0].upper() + cls[1:])
        except ValueError:
            pass

    text_tokens = clip.tokenize(classes).to(device)
    with torch.no_grad():
        class_features_m = model.encode_text(text_tokens)
        class_features_m = (class_features_m / class_features_m.norm(dim=-1, keepdim=True)).cpu().numpy()

    to_delete = set()
    similarities, indices, _ = index.search_and_reconstruct(class_features_m, 512)
    remove_flags = similarities >= sim_cutoff
    for i in range(len(class_features_m)):
        remove_indices = indices[i][remove_flags[i]]
        remove_similarities = similarities[i][remove_flags[i]]
        for rem_index, rem_sim in zip(remove_indices, remove_similarities):
            to_delete.add(concepts[rem_index])
            print(
                "Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(
                    classes[i], concepts[rem_index], rem_sim, concepts[rem_index]
                )
            )
    return to_delete


def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):

    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    concept_features = mpnet_model.encode(concepts)

    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods_self(concepts)

    dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i, j]
            if prod >= sim_cutoff and i != j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    # Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print(
                                "{} - {} , sim:{:.4f} - Deleting {}".format(
                                    concepts[i], concepts[j], dot_prods[i, j], concepts[i]
                                )
                            )
                    else:
                        to_delete.append(j)
                        if to_print:
                            print(
                                "{} - {} , sim:{:.4f} - Deleting {}".format(
                                    concepts[i], concepts[j], dot_prods[i, j], concepts[j]
                                )
                            )

    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def filter_too_similar_faiss(
    concepts, index, model, sim_cutoff, print_prob=0, device="cuda", batchsize=256, sampling_prob=0.01
):
    to_delete = set()
    concepts_np = np.array(concepts)
    sample_nums = int(len(concepts) / batchsize * sampling_prob)
    sample_batches = random.sample(list(range(0, len(concepts), batchsize)), sample_nums)
    for i in tqdm(sample_batches):
        batch_concepts = concepts_np[i : i + batchsize]
        text_tokens = clip.tokenize(batch_concepts, truncate=True).to(device)
        with torch.no_grad():
            concept_feats = model.encode_text(text_tokens)
            concept_feats = (concept_feats / concept_feats.norm(dim=-1, keepdim=True)).cpu().numpy()
        similarities, indices, embeddings = index.search_and_reconstruct(concept_feats, 8)
        block = np.concatenate([concept_feats, np.concatenate(embeddings)])
        block_score = (block @ block.T).sum(axis=-1)
        query_scores = block_score[: len(concept_feats)]
        key_scores = block_score[len(concept_feats) :].reshape(len(concept_feats), -1)
        remove_flags_key = (similarities >= sim_cutoff) & (key_scores < np.expand_dims(query_scores, -1))
        remove_flags_query = (
            np.sum((similarities >= sim_cutoff) & (np.expand_dims(query_scores, -1) < key_scores), axis=-1) > 1
        )
        to_delete |= set(concepts_np[indices[remove_flags_key]].tolist())
        to_delete |= set(batch_concepts[remove_flags_query].tolist())
        # for j, concept in enumerate(batch_concepts):
        #     if concept in to_delete:
        #         continue
        #     remove_indices = indices[j][remove_flags[j]]
        #     remove_embeddings = embeddings[j][remove_flags[j]]
        #     remove_similarities = similarities[j][remove_flags[j]]
        #     for rem_index, rem_sim, rev_sim in zip(remove_indices, remove_similarities, reverse_similarities):
        #         # Checking self-reference and deprication
        #         if (i + j) == rem_index or concepts[rem_index] in to_delete:
        #             continue
        #         query_score = similarities[j].sum()
        #         key_score = rev_sim.sum()
        #         to_print = random.random() < print_prob
        #         if query_score > key_score:
        #             if to_print:
        #                 print(
        #                     "{} - {}, sim:{:.3f} - Deleting {}".format(
        #                         concepts[rem_index], concept, rem_sim, concepts[rem_index]
        #                     )
        #                 )
        #             to_delete.add(concepts[rem_index])
        #         else:
        #             if to_print:
        #                 print(
        #                     "{} - {}, sim:{:.3f} - Deleting {}".format(concepts[rem_index], concept, rem_sim, concept)
        #                 )
        #             to_delete.add(concept)
        #             break
    return to_delete


def _clip_dot_prods(list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)

    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1) / batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size * i : batch_size * (i + 1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2) / batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size * i : batch_size * (i + 1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)

    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()


def _clip_dot_prods_self(list_, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text = clip.tokenize(list_).to(device)

    features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text) / batch_size))):
            features.append(clip_model.encode_text(text[batch_size * i : batch_size * (i + 1)]))
        features = torch.cat(features, dim=0)
        features /= features.norm(dim=1, keepdim=True)

    dot_prods = features @ features.T
    return dot_prods.cpu().numpy()


def most_similar_concepts(word, concepts, device="cuda"):
    """
    returns most similar words to a given concepts
    """
    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    word_features = mpnet_model.encode([word])
    concept_features = mpnet_model.encode(concepts)

    dot_prods_m = word_features @ concept_features.T
    dot_prods_c = _clip_dot_prods([word], concepts, device)

    dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4
    min_distance, indices = torch.topk(torch.FloatTensor(dot_prods[0]), k=5)
    return [(concepts[indices[i]], min_distance[i]) for i in range(len(min_distance))]
