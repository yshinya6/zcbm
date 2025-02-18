import pdb
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

__all__ = ["RetrievalDatabase"]

RETRIEVAL_DATABASES_URLS = {
    "cc12m": "https://drive.google.com/uc?id=1HyM4mnKSxF0sqzAe-KZL8y-cQWRPiuXn&confirm=t",
    "english_words": "https://drive.google.com/uc?id=197poGaUJVP1Mh1qPL5yaNrYJuvd3JRb-&confirm=t",
    "pmd_top5": "https://drive.google.com/uc?id=15SDIf7KM8VIG_AxdnKkL1ODr_igOuZSD&confirm=t",
    "wordnet": "https://drive.google.com/uc?id=1q_StrVCnj8fPgvghXw-fSxp4qaSe0xvk&confirm=t",
}


class RetrievalDatabase:
    """Retrieval database.

    Args:
        database_dir (str): Path to the index directory.
    """

    def __init__(self, index_path: str, metadata_path: str, use_gpu=False):
        index = (
            faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            if Path(index_path).exists()
            else None
        )
        if use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        with open(metadata_path, "r") as f:
            concepts = [line.strip() for line in f]
        self.concept_array = np.array(concepts)
        self._index = index

    def _map_to_metadata(self, indices: list, distances: list, embs: list, num_concepts: int):
        """Map the indices to metadata.

        Args:
            indices (list): List of indices.
            distances (list): List of distances.
            embs (list): List of results embeddings.
            num_images (int): Number of images.
        """
        metas = self.concept_array[indices]
        output = {}
        output["concepts"] = (metas,)
        output["similarities"] = distances
        output["embeddings"] = embs

        return output

    def query(self, query: np.matrix, num_samples: int = 32) -> list[list[dict]]:
        """Query the database.

        Args:
            query (np.matrix): Query to search.
            modality (str): Modality to search. One of `image` or `text`. Default to `text`.
            num_samples (int): Number of samples to return. Default is 40.
        """
        index = self._index

        similarities, indices, embeddings = index.search_and_reconstruct(query, num_samples)
        total_results = []
        for i, _ in enumerate(similarities):
            results = self._map_to_metadata(indices[i], similarities[i], embeddings[i], num_samples)
            total_results.append(results)

        return total_results
