# Flickr30k Text-to-Image Retrieval

Text-to-image retrieval on Flickr30k (Karpathy split). Five approach families compared:

1. Zero-shot CLIP (ViT-B/32, ViT-B/16)
2. Projection head on frozen CLIP B/32
3. Full CLIP fine-tune
4. LoRA r=8 on `q_proj`/`v_proj`
5. From-scratch dual encoders — ViT or ResNet18+BiLSTM × word or BPE tokenizer

All fine-tuning uses `HardNegativeInfoNCELoss` (in-batch InfoNCE with hard-negative weighting).

## Setup

```bash
uv sync
```

`HF_TOKEN` in `.env` is needed if the gated Flickr30k mirror is used.

## Notebooks

Run in order — earlier notebooks produce artifacts later ones consume:

| Notebook                            | Purpose                                                        |
| ----------------------------------- | -------------------------------------------------------------- |
| `01_eda.ipynb`                      | Dataset EDA + Karpathy split sanity                            |
| `02_clip_baseline_and_errors.ipynb` | Zero-shot CLIP B/32 & B/16, Recall@K, failure categorisation   |
| `03_clip_finetuning.ipynb`          | Projection / Full / LoRA — comparison + hard-negative ablation |
| `04_scratch_dual_encoders.ipynb`    | From-scratch ViT or CNN × word or BPE                          |
| `05_analysis.ipynb`                 | Read-only cross-model comparison                               |

Execute a notebook in-place:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/02_clip_baseline_and_errors.ipynb
```

WandB tracking (project `nsiete-flickr30k-clip`) is required.

## Evaluation Metrics

- **Recall@K** (K=1, 5, 10) — fraction of queries with ground-truth image in top-K
- **Median Rank**, **Mean Rank** of ground-truth across queries

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)
