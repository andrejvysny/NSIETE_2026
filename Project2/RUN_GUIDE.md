# RUN_GUIDE — running all experiments on pectra (RTX 5090)

Operational guide. Not a submission doc. Assumes the project tree is at `~/Project2/` on pectra and that you have wandb credentials set up.

**Pre-flight status (already done by setup automation, you can skip step 0 if you know these are true):**

- ✅ Project tree synced via rsync (~4.7 GB at `~/Project2/`)
- ✅ `uv sync` completed — `.venv/` populated with torch 2.11.0+cu130, transformers, peft, wandb, etc.
- ✅ Torch sees the RTX 5090: `torch.cuda.is_available() == True`
- ✅ `scripts/smoke_test_scratch.py` passes on CUDA — all 4 from-scratch variants build, forward + backward + L2-norm checks all green
- ⏳ **You still need to log into wandb interactively** (one-time, see step 0.3)

---

## 0. One-time setup (skip 0.1 + 0.2 if pre-flight is green above)

```bash
ssh pectra
cd ~/Project2
```

### 0.1 Make uv visible

`~/.bashrc` already exports `~/.local/bin` to PATH and aliases `python`/`python3` to `uv run python`. For non-interactive ssh sessions, source it manually:

```bash
source ~/.bashrc
# OR equivalently:
export PATH="$HOME/.local/bin:$PATH"
```

### 0.2 Install / refresh dependencies

```bash
uv sync                         # creates / updates .venv from uv.lock
```

If pectra's CUDA wheel index needs explicit selection (rarely needed on this box):

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124
```

### 0.3 WandB login (interactive — required, not yet done on pectra)

```bash
uv run wandb login              # paste API key from https://wandb.ai/authorize
```

Key is persisted to `~/.netrc` so subsequent runs don't re-prompt. Without this, training notebooks will still run but wandb will print warnings and skip remote logging.

### 0.4 Confirm CUDA visible

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
# expected: cuda: True NVIDIA GeForce RTX 5090
```

If `uv sync` complains about disk space or pulls torch with CPU wheels, force the CUDA index:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124
```

(replace `cu124` with whatever matches the system CUDA — check `nvidia-smi`).

---

## 1. Verify the from-scratch pipeline before any training

```bash
uv run python scripts/smoke_test_scratch.py
```

Expected output: `All variants OK` plus param counts (~5-27M each). If this fails, fix before running notebooks.

---

## 2. Notebook execution order

Each notebook ends with `finish_wandb()` so wandb runs are properly closed. All
training writes:

- `data/models/<run_name>/` — checkpoint(s) + tokenizer (for scratch runs)
- `data/embeddings/<run_name>_{images,texts}_test.npy` — test embeddings for the app + bidirectional eval
- `data/results/<run_name>.json` — Recall@K + MedR + MeanR
- WandB run under project `nsiete-flickr30k-clip` (system metrics, gradients, train+val curves, test summary, model artifact)

### Recommended order on pectra

| #   | Notebook                       | What it does                              | Wall-clock (5090) | Already done? |
| --- | ------------------------------ | ----------------------------------------- | ----------------- | ------------- |
| 1   | `01_eda.ipynb`                 | EDA on Flickr30k                          | <2 min            | ✅ (no GPU)   |
| 2   | `02_clip_baseline.ipynb`       | Zero-shot CLIP B/32 + B/16                | ~5 min            | ✅            |
| 3   | `03_error_analysis.ipynb`      | Failure categorization on baseline        | <2 min            | ✅            |
| 4   | `04_finetune_projection.ipynb` | Frozen CLIP + 256d projection heads       | ~5 min            | ✅            |
| 5   | `06_finetune_lora.ipynb`       | LoRA r=8 on q_proj/v_proj                 | ~30-60 min        | ⏳ rerun      |
| 6   | `05_finetune_full.ipynb`       | Full-CLIP fine-tune + forgetting analysis | ~45-90 min        | ⏳            |
| 7   | `07_scratch_vit.ipynb`         | From-scratch ViT + Transformer (2 toks)   | ~3-6 h total      | ⏳            |
| 8   | `08_scratch_cnn.ipynb`         | From-scratch ResNet18 + Bi-LSTM (2 toks)  | ~3-6 h total      | ⏳            |

NB 01-04 can be skipped on pectra if their artifacts already synced (check `data/models/projection_head/best_projection.pt` and `data/results/baseline_*.json` after rsync). Re-running is cheap and avoids any cross-machine drift.

### Headless execution (preferred)

Each notebook re-runs in place via `nbconvert`. Use `tee` to capture logs and `EPOCH_OVERRIDE=N` for quick smoke runs:

```bash
# Single notebook, full run, log to file
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    notebooks/07_scratch_vit.ipynb 2>&1 | tee notebooks/07_run.log

# Quick 2-epoch sanity check before committing to 30 epochs
EPOCH_OVERRIDE=2 uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    notebooks/07_scratch_vit.ipynb 2>&1 | tee notebooks/07_smoke.log
```

### Run all training back-to-back (background, screen/tmux recommended)

```bash
# Start a tmux session so it survives ssh disconnects
tmux new -s train

# Inside tmux:
cd ~/Project2
for nb in notebooks/06_finetune_lora.ipynb notebooks/05_finetune_full.ipynb \
          notebooks/07_scratch_vit.ipynb notebooks/08_scratch_cnn.ipynb; do
    echo "=== running $nb ==="
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee "${nb%.ipynb}_run.log"
done
echo "=== all done ==="

# Detach from tmux: Ctrl+b then d
# Reattach later:  tmux attach -t train
```

Total expected wall-clock: **8-15 hours** for the full sweep on a 5090.

### Final consolidated comparison

After all training notebooks finish, re-run NB 06 cells 16+17 (or just open and execute) to refresh `data/results/final_comparison.json` and `final_comparison_bidirectional.json` with all 9 rows.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    notebooks/06_finetune_lora.ipynb 2>&1 | tee notebooks/06_compare.log
```

(this re-runs LoRA training too — set `RUN_TRAINING = False` and `TRAINING_COMPLETE = True` in cells 12 and 16/17/18/19 if you only want the comparison block, OR comment out the training cell)

---

## 3. WandB — what gets tracked, where to look

Project URL: `https://wandb.ai/<your-entity>/nsiete-flickr30k-clip`

Per-run logged automatically:

- **Hyperparams (config):** approach, model, batch_size, lr, num_epochs, weight_decay, temperature, hard_negative_weight, loss, seed, device, plus arch-specific keys (lora_r/lora_target/projection_dim/joint_dim/vocab_size/...)
- **Train metrics (per step):** loss, lr, in-batch acc_i2t, acc_t2i
- **Val metrics (per epoch):** R@1, R@5, R@10, MedianR, MeanR, epoch_loss
- **Test summary (sticky):** test/R@1 ... test/MeanR (visible in run sidebar)
- **System metrics (auto):** GPU util/temp/memory, CPU util, disk, network
- **Model artifacts:** small checkpoints uploaded (projection ~3 MB, LoRA ~2 MB, scratch ~50 MB). Full FT (~600 MB) is NOT uploaded — too big.
- **Gradients:** logged for full-FT, LoRA, and scratch runs via `wandb.watch(model, log="gradients", log_freq=200)` — useful for catching divergence

Tags identify runs: `["full_ft"]`, `["lora","r=8"]`, `["projection_head"]`, `["scratch","vit","word"]`, etc.

To compare runs in the wandb UI: select all from the runs list, then Workspace → Auto-grouped panels show side-by-side R@K curves.

For ablations (NB 06 cell 25), set `RUN_ABLATION = True` to run hard-neg-weight ∈ {0, 1, 2, 4} as 4 wandb runs tagged `["ablation","hardneg_w"]`.

---

## 4. Sync results back to laptop (after training)

From the **laptop**, pull only the small deliverables — skip the large flickr30k cache and any new venv content on pectra:

```bash
rsync -avz --progress \
    --exclude='.venv/' --exclude='wandb/' --exclude='__pycache__/' \
    --exclude='data/flickr30k/' \
    pectra:Project2/data/ \
    /Users/andrejvysny/fiit/nsiete/NSIETE_2026/Project2/data/

rsync -avz --progress \
    pectra:Project2/notebooks/ \
    /Users/andrejvysny/fiit/nsiete/NSIETE_2026/Project2/notebooks/
```

This pulls back:

- `data/models/{lora_clip,full_finetune,scratch_*}/` — all checkpoints
- `data/embeddings/{lora,full,scratch_*}_b32_*_test.npy` — for the app
- `data/results/*.json` — including `final_comparison.json`
- All notebooks (with executed outputs / loss curves / comparison tables)

**Skip the full-FT 600 MB checkpoint if you don't want it locally:**

```bash
--exclude='data/models/full_finetune/'
```

then drive the app from cached embeddings only (Run Evaluation page works; live encoding for `full_b32` will fail — acceptable).

---

## 5. Streamlit demo (laptop, after sync)

```bash
cd /Users/andrejvysny/fiit/nsiete/NSIETE_2026/Project2
uv run streamlit run app.py
```

The dropdown auto-discovers all 9 models (5 CLIP + 4 scratch) once their checkpoints + cached test embeddings are present.

---

## 6. Known gotchas

- **`uv run wandb login`** — `wandb` is the venv binary; bare `wandb login` won't be on PATH unless venv is activated.
- **NB 06 partial state:** the local M4 run wrote a LoRA adapter but the eval cell errored on the old `pooler_output` issue (now fixed in `src/training.py:341-345`). On pectra, NB 06 will train cleanly from scratch — no recovery needed.
- **AMP with MPS:** the scratch notebooks set `scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")` so on MPS or CPU the scaler is a no-op and training falls back to fp32. AMP only kicks in on pectra's CUDA.
- **HF dataset re-download:** if `data/flickr30k/` was synced, `load_karpathy_splits` finds the local Arrow files and skips download. If you want a fresh download from HF Hub, delete that folder first.
- **`EPOCH_OVERRIDE` env var:** both scratch notebooks read it — useful for fast iteration. e.g. `EPOCH_OVERRIDE=2` runs 2 epochs instead of 30.
- **GPU memory:** all scratch models fit in ~6 GB at BS=128. If OOM (e.g. shared 5090), drop BATCH_SIZE in the hyperparams cell. The 5090 has 32 GB so this should never trigger.

---

## 7. Quick reference — full reproduction from clean clone

```bash
ssh pectra
cd ~/Project2 && uv sync && uv run wandb login
uv run python scripts/smoke_test_scratch.py
tmux new -s train
for nb in notebooks/02_clip_baseline.ipynb notebooks/03_error_analysis.ipynb \
          notebooks/04_finetune_projection.ipynb notebooks/06_finetune_lora.ipynb \
          notebooks/05_finetune_full.ipynb notebooks/07_scratch_vit.ipynb \
          notebooks/08_scratch_cnn.ipynb; do
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "$nb" 2>&1 | tee "${nb%.ipynb}_run.log"
done
```

Detach from tmux, come back ~12 hours later, sync results to laptop.
