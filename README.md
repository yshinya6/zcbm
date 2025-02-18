# [Zero-shot Concept Bottleneck Models](https://arxiv.org/abs/2502.09018)
We introduce an interpretable model family called zero-shot concept bottleneck models (Z-CBMs), which predict concepts and labels in a fully zero-shot manner without training neural networks. Z-CBMs utilize a large-scale concept bank, which is composed of millions of vocabulary extracted from the web, to describe arbitrary input in various domains. For the input-to-concept mapping, we introduce concept retrieval, which dynamically finds input-related concepts by the cross-modal search on the concept bank. In the concept-to-label inference, we apply concept regression to select essential concepts from the retrieved concepts by sparse linear regression.

![image](https://github.com/user-attachments/assets/75b117ac-a94d-4389-8b5b-09610177e89d)


## Requirements
- An NVIDIA GPU (we used an A100 with 80 GB VRAM)
- Python Libraries: See `requirements.txt`

## Preparation
### Concept collection from existing caption datasets
Z-CBMs use the captions in the following datasets:
- [Flickr30k](https://github.com/awsaf49/flickr-dataset)
- [CC3M](https://ai.google.com/research/ConceptualCaptions/download) (required only metadata containing captions)
- [CC12M](https://storage.googleapis.com/conceptual_12m/cc12m.tsv) (required only metadata containing captions)
- [YFCC15M](https://huggingface.co/datasets/mehdidc/yfcc15m) (required only metadata containing captions)

After download the metadata, put them into `metadata` directory and run `python concept_collector.py` for each dataset as follows.

```bash
python concept_collector.py --dataset cc3m --metadata_path ./metadata/CC3M/Image_Labels_Subset_Train_GCC-Labels-training.tsv
```

### Concept bank construction
[2025/2/18] We are preparing to provide pre-computed faiss indices. Stay tuned!

```bash
python concept_construction.py --model "ViT-B/32" --base_concepts base_concepts/{flickr30k,cc3m,cc12m,yfcc15m}.yaml --filtering_similar --use_faiss_gpu
```

## Run the experiments
### Zero-shot Inference

```bash
python main/inference.py --config_path configs/01_multiple_dataset/imagenet.yaml
```

## Citation
```
@article{yamaguchi_2025_ZCBM,
  title={Zero-shot Concept Bottleneck Models},
  author={Yamaguchi, Shin'ya and Nishida, Kosuke and Chijiwa, Daiki and Ida, Yasutoshi},
  journal={arXiv preprint arXiv:2502.09018},
  year={2025}
}
```
