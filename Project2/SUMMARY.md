# SUMMARY — Flickr30k Text-to-Image Retrieval

NSIETE Project 2. Five experiment families compared on the Flickr30k Karpathy test split (1 000 images, 5 000 captions). Retrieval is cosine similarity in a shared L2-normalised embedding space. Every fine-tune uses `HardNegativeInfoNCELoss` (in-batch InfoNCE with hard-negative weighting=2.0).

## Headline numbers (text-to-image, Karpathy test)

| #   | Approach                          | R@1    | R@5    | R@10   | MedR | MeanR | Trainable params |
| --- | --------------------------------- | ------ | ------ | ------ | ---- | ----- | ---------------- |
| 1   | CLIP ViT-B/32 zero-shot           | 58.6 % | 83.5 % | 89.9 % | 1    | 6.05  | 0                |
| 2   | CLIP ViT-B/16 zero-shot           | 62.1 % | 85.6 % | 92.0 % | 1    | 4.95  | 0                |
| 3   | Projection head on B/32           | 63.8 % | 88.1 % | 93.4 % | 1    | 4.69  | 788 K (0.5 %)    |
| 4   | LoRA r=8 on B/32                  | 67.2 % | 89.2 % | 94.2 % | 1    | 4.05  | 491 K (0.32 %)   |
| 5   | Full FT B/32                      | 68.4 % | 90.0 % | 94.4 % | 1    | 3.90  | 151 M (100 %)    |
| 6   | Scratch ResNet18 + Bi-LSTM (word) | 9.9 %  | 27.1 % | 37.9 % | 21   | 102.4 | 16.5 M           |
| 7   | Scratch ResNet18 + Bi-LSTM (BPE)  | 9.2 %  | 25.5 % | 37.0 % | 21   | 86.2  | 26.7 M           |
| 8   | Scratch ViT + Transformer (BPE)   | 5.5 %  | 17.7 % | 26.1 % | 43   | 124.1 | 14.2 M           |
| 9   | Scratch ViT + Transformer (word)  | 5.1 %  | 15.5 % | 23.9 % | 47   | 127.0 | 6.6 M            |

Random baseline: R@1 = 0.001 (1 / 1 000), R@5 = 0.005, R@10 = 0.010.

## Bidirectional retrieval

Image-to-text scores higher than text-to-image across every CLIP-family approach, because each image has 5 valid captions in the gallery and one hit suffices for R@1. The gap is roughly 20 pp on the zero-shot baselines and roughly 15 pp on the fine-tuned variants; fine-tuning narrows the asymmetry without removing it.

| Approach   | t2i R@1 | i2t R@1 | Δ     |
| ---------- | ------- | ------- | ----- |
| CLIP B/32  | 58.6 %  | 79.1 %  | +20.5 |
| CLIP B/16  | 62.1 %  | 82.7 %  | +20.6 |
| Projection | 63.8 %  | 78.5 %  | +14.7 |
| LoRA       | 67.2 %  | 82.1 %  | +14.9 |
| Full FT    | 68.4 %  | 84.2 %  | +15.8 |

The relative ordering of approaches is identical in both directions, so the t2i numbers are sufficient for the comparison.

## Methodology

The Karpathy split (29 000 / 1 014 / 1 000 images, image-disjoint) is used throughout. Frozen CLIP B/32 and B/16 are evaluated zero-shot. Three fine-tuning recipes are applied to B/32: a 256-d MLP projection head trained on cached frozen CLIP features (3 epochs, LR=1e-4); a full fine-tune of all 151 M parameters (3 epochs, LR=2e-6); and LoRA r=8 adapters on every `q_proj`/`v_proj` (10 epochs, LR=1e-4). Four from-scratch dual encoders are trained from random initialisation: ViT-small or ResNet18 image tower, paired with Transformer or Bi-LSTM text tower, each variant trained with both a word and a BPE tokenizer. Common settings for the scratch runs are batch size 256, AdamW with cosine schedule, bf16 AMP, and `EarlyStopping(patience=10, max_epochs=150)` on val R@1. All training runs on a single RTX 5090. The evaluation pipeline is identical across approaches: encode test images and captions, compute cosine similarity, report ranking metrics.

## Key findings

LoRA recovers about 88 % of the full-FT lift at 0.3 % of the trainable parameter count. Full FT lifts R@1 by 9.8 pp (58.6 to 68.4) at 151 M parameters and a 1.7 GB checkpoint; LoRA r=8 lifts R@1 by 8.6 pp (58.6 to 67.2) at 491 K parameters and an under-2 MB adapter. The 1.2 pp residual gap is below typical seed noise on this benchmark, so LoRA dominates in any setting where adapter size or training cost is a real constraint.

The projection head is a strong cheap baseline: 5.2 pp R@1 (58.6 to 63.8) at 788 K parameters and roughly three minutes of training on cached embeddings. It is useful as an "is fine-tuning worth it?" probe before committing to full FT or LoRA.

The CLIP pretraining gap is roughly 50 pp R@1. The best from-scratch model (CNN+word) reaches 9.9 % R@1; CLIP B/16 zero-shot reaches 62.1 %. Five orders of magnitude more pretraining data (29 K Flickr30k pairs versus the 400 M web pairs of CLIP) buys that gap. This is the central number of the project: the cost-benefit case for using pretrained encoders rather than training a retrieval model from scratch on whatever in-house data is available.

CNN beats ViT from random initialisation at this data scale. Both CNN variants land near 9.5 % R@1 and both ViT variants near 5.3 %. The 4–5 pp gap reflects the convolutional inductive bias (locality, translation equivariance) at 29 K images. ViTs need scale to win, and we do not have it; CLIP picks ViTs precisely because it does.

Tokenizer choice is sub-1pp noise at this data scale. ViT prefers BPE by 0.4 pp, CNN prefers word by 0.7 pp, and both are within the noise floor for a 1 000-image test set with a single seed. BPE's expected advantage (no `<unk>`, subword fallback for rare words) does not appear here because Flickr30k captions are short and the long-tail vocabulary that BPE rescues is small.

Most CLIP failures are near misses, and this is exactly the regime that hard-negative weighting targets. About 87 % of B/32 zero-shot failures have a similarity gap below 0.05 between the GT image and the top-1 retrieved. The failure-mode breakdown is counting (28.6 %), other (22.9 %), colour (22.1 %), action (13.9 %), spatial (6.5 %), attribute (6.0 %). Counting and colour together account for half the failures; both are vision-language priors that CLIP learned weakly during pretraining. Fine-tuning on Flickr30k re-shapes the embedding to disambiguate near-miss cases but cannot add new vision knowledge.

Catastrophic forgetting after full FT is mild on a 30-query generic probe. Median rank shift is 1, mean is 7.77, max is 76, and 43 % of queries had unchanged top-1. This is driven by the conservative recipe (LR=2e-6, 3 epochs); a higher LR or longer training would worsen the picture. That trade-off is the standard argument for parameter-efficient fine-tuning over full FT in production. LoRA has zero forgetting on this probe by construction, since the original weights are bit-exact unchanged.

Early stopping was the load-bearing change for the from-scratch experiments. Replacing the fixed 30-epoch budget with `EarlyStopping(patience=10, max_epochs=150)` on val R@1 lifted every scratch variant by 1.5–3 pp (CNN+word: 6.9 % to 9.9 %). Best epochs landed between 43 and 68, all four variants stopped well under the 150 ceiling, and the takeaway is that letting validation drive the training budget is preferable to a fixed epoch count.

## Notebook map

| #   | Notebook                            | Purpose                                         | Key outputs                                                                      |
| --- | ----------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------- |
| 01  | `01_eda.ipynb`                      | Dataset stats and Karpathy split sanity         | Image/caption distributions, vocab size 20 320, 5 captions/image                 |
| 02  | `02_clip_baseline_and_errors.ipynb` | Zero-shot CLIP B/32 and B/16, failure analysis  | `data/results/baseline_b{32,16}.json`, failure-category bar chart                |
| 03  | `03_clip_finetuning.ipynb`          | Projection / Full FT / LoRA + hard-neg ablation | `data/results/{projection_head,full_finetune,lora}_b32.json`, forgetting summary |
| 04  | `04_scratch_dual_encoders.ipynb`    | Four from-scratch dual encoders (early-stopped) | `data/results/scratch_{vit,cnn}_{word,bpe}.json`                                 |
| 05  | `05_analysis.ipynb`                 | Cross-model aggregator                          | `data/results/final_comparison*.json`, comparison PNG                            |

Execution order is strict: 02 produces baseline embeddings that 03 reuses; 03 writes the comparison file that 05 reads alongside 04's outputs.

## Repository layout (submission scope)

```
Project2/
├── README.md
├── CLAUDE.md            # working notes for code-assistant context
├── SUMMARY.md           # this file
├── pyproject.toml       # uv-managed deps (Python 3.12)
├── uv.lock
├── src/
│   ├── config.py        # paths, model registry, get_device()
│   ├── data.py          # Karpathy split loader
│   ├── clip_embeddings.py
│   ├── training.py      # HardNegativeInfoNCELoss, EarlyStopping, training_step, run_validation
│   ├── evaluation.py    # evaluate_text_to_image / save_results
│   ├── retrieval.py     # NumPy similarity helpers
│   ├── scratch_model.py # build_vit_dual / build_cnn_dual
│   ├── scratch_tokenizer.py  # WordTokenizer + ClipBPETokenizer
│   └── visualize.py
├── notebooks/   (5 ipynb files, outputs embedded)
└── data/
    └── results/         # 12 JSON + 2 PNG  (metric files included with the submission)
```

The directories `data/{flickr30k,embeddings,models}/` and `notebooks/wandb/` are gitignored, since they are too large for a submission zip. Scratch models can be reproduced from `notebooks/04_scratch_dual_encoders.ipynb` and the cached HF Flickr30k dataset; CLIP weights are downloaded from HuggingFace on first run.

## Reproduction

```bash
uv sync                               # install deps from uv.lock
# (optional) HF_TOKEN in .env if the gated Flickr30k mirror is used

# Re-run any notebook end-to-end:
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/04_scratch_dual_encoders.ipynb

# Smoke-mode (3 epochs each variant, ~10 min on RTX 5090):
EPOCH_OVERRIDE=3 uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/04_scratch_dual_encoders.ipynb
```

All training ran on Pectra (RTX 5090, 32 GB), since the local Mac has no NVIDIA GPU. The full pipeline (smoke runs aside) takes about an hour of wall-clock time for 02 + 03 + 04 + 05 on Pectra.

## Honest limitations

Every variant is trained with a single seed, so sub-1pp differences (full FT vs LoRA, word vs BPE) should be read with that caveat. The 1 000-image test set means R@1 differences below roughly 1 pp are inside noise. LoRA rank is not swept (r=8 only) and the projection-head architecture is not swept. The full-FT recipe is conservative (LR=2e-6, 3 epochs); a higher LR with proper warm-up would likely gain 0.5–1 pp at the cost of a worse forgetting profile. Catastrophic forgetting is evaluated on 30 hand-curated queries, not a held-out general-domain benchmark. For the from-scratch runs, `EarlyStopping(patience=10)` is a deliberate but unswept choice; tighter or looser patience would shift numbers slightly in either direction. Hard-negative weighting is fixed at 2.0 (per the loss design), and the ablation in notebook 03 is on the projection head only.

## References

Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). arXiv:2103.00020.

Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Faghri et al. (2018). VSE++: Improving Visual-Semantic Embeddings with Hard Negatives. arXiv:1707.05612.

Karpathy & Fei-Fei (2015). Deep Visual-Semantic Alignments for Generating Image Descriptions (Karpathy split definition). CVPR.
