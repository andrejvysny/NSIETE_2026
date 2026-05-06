# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Flickr30k text-to-image retrieval. Five experiment families compared on the Karpathy test split:

1. **Zero-shot CLIP** (ViT-B/32, ViT-B/16)
2. **Projection head** on frozen CLIP B/32 features
3. **Full fine-tune** of CLIP B/32
4. **LoRA r=8** on CLIP B/32 (`q_proj`, `v_proj`)
5. **From-scratch dual encoders** (ViT or ResNet18+BiLSTM) × (word or BPE tokenizer)

All retrieval is cosine similarity in a shared L2-normalized embedding space. Every fine-tune uses `HardNegativeInfoNCELoss` (in-batch InfoNCE with hard-negative weighting).

Sub-project of the broader NSIETE course repo. Has its own venv, deps, and conventions — the parent `CLAUDE.md` (NumPy-only weekly assignments) does NOT apply here.

## Environment & Commands

`uv` is the runtime/package manager. Python 3.12 (see `.python-version`). NEVER use plain `pip`/`python` — always prefix with `uv run`.

```bash
uv sync                                       # install deps from uv.lock
uv run ruff check . && uv run ruff format .   # lint + format
```

Execute a notebook in-place (primary training workflow — every training run is a notebook):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/04_scratch_dual_encoders.ipynb
```

`EPOCH_OVERRIDE=N` env var is read by the training notebooks for fast smoke runs.

## Notebooks

Strict pipeline — earlier notebooks produce artifacts later ones consume.

| #   | Notebook                            | Purpose                                                                   |
| --- | ----------------------------------- | ------------------------------------------------------------------------- |
| 01  | `01_eda.ipynb`                      | EDA, Karpathy split sanity                                                |
| 02  | `02_clip_baseline_and_errors.ipynb` | Zero-shot CLIP B/32 + B/16, Recall@K, failure categorisation              |
| 03  | `03_clip_finetuning.ipynb`          | Projection head + full FT + LoRA r=8, comparison + hard-negative ablation |
| 04  | `04_scratch_dual_encoders.ipynb`    | From-scratch ViT and ResNet18+BiLSTM × word/BPE tokenizers                |
| 05  | `05_analysis.ipynb`                 | Read-only cross-model comparison + figures                                |

Every training notebook writes to:

- `data/models/<run_name>/` — checkpoint(s) + tokenizer (scratch only)
- `data/embeddings/<run_name>_{images,texts}_test.npy` — L2-normalized test embeddings
- `data/results/<run_name>.json` — Recall@K, MedR, MeanR
- WandB project `nsiete-flickr30k-clip`, run tagged with approach (WandB tracking is required)

## Architecture

### `src/` — library code (no scripts, no app)

- **`config.py`** — paths (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `EMBEDDINGS_DIR`, `RESULTS_DIR`), `CLIP_MODELS` dict, `WANDB_PROJECT`, `get_device()` (cuda → mps → cpu), `set_seeds(42)`.
- **`data.py`** — Flickr30k loader. `load_karpathy_splits()` is the canonical entry point (~29K/1K/1K). `restval` is merged into train. `get_all_captions_flat()` returns `(captions, gt_indices)` for retrieval eval.
- **`clip_embeddings.py`** — `load_clip_model(key)` and `load_finetuned_model(name)`. The latter is a model-name registry: `clip-vit-b-32`, `clip-vit-b-16`, `projection_b32`, `lora_b32`, `full_b32`, `scratch_{vit,cnn}_{word,bpe}`. All returned models expose `get_image_features` and `get_text_features`. `encode_images` / `encode_texts` always L2-normalize. Some HF CLIP variants return a `BaseModelOutputWithPooling` instead of a tensor — these are unwrapped via `pooler_output` in both encoding helpers and `training_step`.
- **`training.py`** — datasets, `HardNegativeInfoNCELoss`, training step, validation, checkpoint I/O, WandB wrappers, `mine_hard_negatives_offline` (qualitative figure only). `init_wandb` / `log_*` / `finish_wandb` are the only WandB touchpoints — notebooks should never `import wandb` directly.
- **`evaluation.py`** — `evaluate_text_to_image(text_emb, image_emb, gt)` and the symmetric `evaluate_image_to_text` for bidirectional analysis. `save_results(results, name)` writes to `data/results/<name>.json`.
- **`retrieval.py`** — pure NumPy similarity ranking helpers.
- **`scratch_model.py`** — `DualEncoder(image_encoder, text_encoder, joint_dim=256)` with builders `build_vit_dual()` and `build_cnn_dual()`. Critical: the API mirrors HF CLIP (`get_image_features`/`get_text_features`) so `training_step`, `encode_images`, `encode_texts` work without branching.
- **`scratch_tokenizer.py`** — `WordTokenizer` (vocab from train captions) and `ClipBPETokenizer` (CLIP's BPE merge table, no pretrained embeddings). Both expose `.tokenize(texts) → {input_ids, attention_mask}`. `save(dir)` / `load_tokenizer(name)` for round-trip.
- **`visualize.py`** — Plotly/matplotlib helpers for notebooks.

### Conventions

- **Embeddings are always L2-normalized** at the encoder boundary. Cosine similarity is just a dot product downstream.
- **Captions are flattened** to `(caption_text, gt_image_idx)` pairs; ground truth is the image's index in the dataset, not its filename.
- **Tensor shape conventions follow PyTorch** (batch first), unlike the parent course repo's NumPy code.
- **`src/training.py` is tightly coupled to the HF CLIP API** — when adding a new model variant, conform to `get_image_features`/`get_text_features` rather than introducing a new code path.
- **WandB logging is namespaced**: `train/*` per step, `val/*` per epoch, `test/*` in run summary. `wandb.watch(model, log="gradients", log_freq=200)` is wired for full FT, LoRA, and scratch.
- Ruff: `line-length=100`, ignores `E501`, `B008`, `UP007` (keeps `Optional[X]` over `X | None`).

### Data layout (gitignored except `data/results/*.json|*.png`)

```
data/
├── flickr30k/              # HF dataset cache (Arrow files); 31,783 images
├── dataset_flickr30k.json  # Karpathy split file
├── models/<run_name>/      # checkpoints (+ tokenizer.json for scratch runs)
├── embeddings/             # cached (N, D) float32 matrices, all L2-normalized
└── results/                # *.json metric reports + comparison PNG
```

`load_karpathy_splits()` lazily downloads both Flickr30k (HF, needs `HF_TOKEN` in `.env` for gated mirrors) and the Stanford Karpathy zip if not on disk.

## Gotchas

- **MPS lacks `torch.cuda.amp.GradScaler` semantics** — scratch notebooks set `enabled=device.type=="cuda"` so AMP is a no-op locally.
- **Full-FT checkpoint is ~600 MB** and is NOT uploaded as a WandB artifact (only smaller adapters/projections are).
- **PEFT `PeftModel.get_image_features` returns `BaseModelOutputWithPooling`**, not a tensor — encoding helpers + `training_step` all unwrap via `if hasattr(x, "pooler_output")`.
