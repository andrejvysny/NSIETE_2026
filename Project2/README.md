# Flickr30k Text-to-Image Retrieval

Text-to-image retrieval system using CLIP on the Flickr30k dataset. Given a text query, retrieves the most relevant images from the gallery using cosine similarity in CLIP's joint embedding space.

## Approach

1. **Baseline**: Zero-shot CLIP (ViT-B/32 and ViT-B/16)
2. **EDA**: Dataset exploration, image/caption analysis, Karpathy split validation
3. **Error Analysis**: Categorize baseline failures, identify improvable error types
4. **Fine-tuning**: Three approaches to improve retrieval:
   - Projection head (frozen CLIP + trainable MLP)
   - Full CLIP fine-tuning (end-to-end)
   - LoRA adapters (parameter-efficient)

All fine-tuning uses **Hard Negative Mining** (VSE++ style) as the core improvement.

## Setup

```bash
uv sync
```

## Notebooks

Run in order:

| Notebook                       | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `01_eda.ipynb`                 | Dataset EDA + Karpathy split analysis         |
| `02_clip_baseline.ipynb`       | CLIP B/32 + B/16 zero-shot baseline, Recall@K |
| `03_error_analysis.ipynb`      | Failure categorization, hardness analysis     |
| `04_finetune_projection.ipynb` | Frozen CLIP + projection head training        |
| `05_finetune_full.ipynb`       | Full CLIP fine-tuning                         |
| `06_finetune_lora.ipynb`       | LoRA adapter fine-tuning                      |

## Streamlit App

```bash
uv run streamlit run app.py
```

Pages: **Search** (text-to-image) | **Gallery** | **Compare** models | **Evaluation** (Recall@K) | **Explorer** (t-SNE)

## Evaluation Metrics

- **Recall@K** (K=1,5,10): fraction of queries where ground-truth image is in top-K
- **Median Rank**: median rank of ground-truth image across all queries
- **Mean Rank**: mean rank of ground-truth image

## Dataset

[Flickr30k](https://huggingface.co/datasets/lmms-lab/flickr30k) with [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip): ~29K train / 1K val / 1K test.

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)
