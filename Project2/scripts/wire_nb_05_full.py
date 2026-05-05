"""Wire notebook 05 (full FT): seed, wandb, training enabled, embedding cache, expanded forgetting test."""

from __future__ import annotations

from pathlib import Path

from _nb_edit import insert_cell, load_nb, save_nb, set_cell_source

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "05_finetune_full.ipynb"


CELL_1_IMPORTS = """\
import sys
from pathlib import Path

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR, get_device, set_seeds
from src.data import load_karpathy_splits, get_all_captions_flat
from src.clip_embeddings import (
    load_clip_model,
    encode_images,
    encode_texts,
)
from src.evaluation import (
    evaluate_text_to_image,
    save_results,
    load_results,
    print_results_table,
)
from src.training import (
    CLIPRetrievalDataset,
    make_collate_fn,
    HardNegativeInfoNCELoss,
    get_optimizer,
    get_scheduler,
    training_step,
    save_checkpoint,
    load_checkpoint,
    run_validation,
    init_wandb,
    log_train_step,
    log_val_metrics,
    log_summary,
    finish_wandb,
)
from src.visualize import (
    plot_training_curves,
    plot_recall_comparison,
    plot_retrieval_results,
)

set_seeds(42)
device = get_device()

# CUDA performance knobs (see scratch notebooks for details).
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""


CELL_7_DATA = """\
import os

# Full CLIP fine-tune: 88M params, larger activation footprint than scratch.
# bf16 AMP lets BS=128 fit comfortably on a 5090.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128" if device.type == "cuda" else "32"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256" if device.type == "cuda" else "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "12" if device.type == "cuda" else "0"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "4")) if NUM_WORKERS > 0 else None

train_dataset = CLIPRetrievalDataset(splits["train"], processor)
loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=make_collate_fn(processor),
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
    drop_last=True,
)
if NUM_WORKERS > 0 and PREFETCH_FACTOR is not None:
    loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
train_loader = DataLoader(train_dataset, **loader_kwargs)

print(f"Training images: {len(train_dataset)}")
print(f"Batches per epoch: {len(train_loader)}")
print(f"Batch size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}")

sample_batch = next(iter(train_loader))
print(f"\\nSample batch:")
print(f"  pixel_values:  {sample_batch['pixel_values'].shape}")
print(f"  input_ids:     {sample_batch['input_ids'].shape}")
print(f"  attention_mask: {sample_batch['attention_mask'].shape}")
"""


# With BS=128 and bf16 we don't need accumulation
CELL_9_HYPER = """\
NUM_EPOCHS = int(os.environ.get("EPOCH_OVERRIDE", "3"))
LR = 2e-6
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1  # effective batch size = BATCH_SIZE
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
LOG_EVERY = 50

# bf16 AMP (no GradScaler needed). Blackwell has native bf16 tensor cores.
AMP_DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32
USE_AMP = device.type == "cuda"

loss_fn = HardNegativeInfoNCELoss(temperature=0.07)
optimizer = get_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)

num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
scheduler = get_scheduler(optimizer, num_training_steps, warmup_ratio=WARMUP_RATIO)

print(f"NUM_EPOCHS={NUM_EPOCHS} BATCH_SIZE={BATCH_SIZE} effective={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Total optimization steps: {num_training_steps}")
print(f"Warmup steps: {int(num_training_steps * WARMUP_RATIO)}")
print(f"AMP={USE_AMP} ({AMP_DTYPE})")
"""


CELL_11_TRAIN = """\
RUN_TRAINING = True  # Set to False to skip training

if RUN_TRAINING:
    CHECKPOINT_DIR = MODELS_DIR / "full_finetune"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    run = init_wandb(
        run_name="full_ft_b32",
        config={
            "approach": "full_finetune",
            "model": "clip-vit-b-32",
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "temperature": 0.07,
            "hard_negative_weight": 2.0,
            "loss": "HardNegativeInfoNCE",
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "seed": 42,
            "device": str(device),
        },
        tags=["full_ft"],
    )
    try:
        import wandb
        wandb.watch(model, log="gradients", log_freq=200)
    except Exception:
        pass

    best_val_r1 = 0.0
    train_losses: list[float] = []
    val_metrics_history: dict[str, list[float]] = {"R@1": [], "R@5": [], "R@10": []}
    step_metrics: list[dict] = []
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses: list[float] = []
        ep_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            with torch.amp.autocast(
                device_type=device.type, enabled=USE_AMP, dtype=AMP_DTYPE
            ):
                loss, metrics = training_step(model, batch, loss_fn, device)

            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            epoch_losses.append(loss.item())
            train_losses.append(loss.item())

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_train_step(
                    {
                        "loss": metrics["loss"],
                        "acc_i2t": metrics["acc_i2t"],
                        "acc_t2i": metrics["acc_t2i"],
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            if (batch_idx + 1) % LOG_EVERY == 0:
                lr_current = scheduler.get_last_lr()[0]
                step_metrics.append({"step": global_step, **metrics, "lr": lr_current})

        # Flush remaining gradients at epoch end
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = float(np.mean(epoch_losses))

        # -- Validation --
        val_results = run_validation(
            model, processor, splits["val"], val_captions, val_gt, device=device
        )

        for k in ["R@1", "R@5", "R@10"]:
            val_metrics_history[k].append(val_results[k])

        log_val_metrics(
            {
                "R@1": val_results["R@1"],
                "R@5": val_results["R@5"],
                "R@10": val_results["R@10"],
                "MedianR": val_results["MedianR"],
                "MeanR": val_results["MeanR"],
                "epoch_loss": avg_loss,
            },
            epoch=epoch,
        )

        print(
            f"Epoch {epoch+1} done in {time.time() - ep_start:.0f}s: loss={avg_loss:.4f} | "
            f"Val R@1={val_results['R@1']:.3f} R@5={val_results['R@5']:.3f} R@10={val_results['R@10']:.3f}",
            flush=True,
        )

        if val_results["R@1"] > best_val_r1:
            best_val_r1 = val_results["R@1"]
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                CHECKPOINT_DIR / "best_full_finetune.pt",
            )
            print(f"  -> New best! Saved checkpoint (R@1={best_val_r1:.3f})")

    print(f"\\nBest validation R@1: {best_val_r1:.3f}")
"""


CELL_13_TEST_EVAL = """\
TRAINING_COMPLETE = True  # Set to False to skip evaluation

if TRAINING_COMPLETE:
    CHECKPOINT_DIR = MODELS_DIR / "full_finetune"
    checkpoint = load_checkpoint(
        CHECKPOINT_DIR / "best_full_finetune.pt", model, optimizer, device=device
    )
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    model.eval()
    with torch.no_grad():
        test_img_emb = encode_images(splits["test"], model, processor, device=device)
        test_txt_emb = encode_texts(test_captions, model, processor, device=device)

    np.save(EMBEDDINGS_DIR / "full_b32_images_test.npy", test_img_emb)
    np.save(EMBEDDINGS_DIR / "full_b32_texts_test.npy", test_txt_emb)
    print(
        f"Cached fully-fine-tuned test embeddings: "
        f"{test_img_emb.shape} images, {test_txt_emb.shape} texts"
    )

    full_ft_results = evaluate_text_to_image(test_txt_emb, test_img_emb, test_gt)
    save_results(full_ft_results, "full_finetune_b32")
    print("\\nFull Fine-tune Results:")
    for k, v in full_ft_results.items():
        print(f"  {k}: {v:.1%}" if k.startswith("R@") else f"  {k}: {v:.4f}")

    log_summary(full_ft_results, prefix="test")
    # Skip artifact upload: full FT checkpoint is ~600 MB
"""


CELL_14_COMPARE = """\
TRAINING_COMPLETE = True  # Set to False to skip comparison

if TRAINING_COMPLETE:
    comparison: dict = {}

    for display, candidates in [
        ("CLIP B/32 (baseline)", ["baseline_b32", "baseline_clip-vit-b-32"]),
        ("CLIP B/32 + Projection", ["projection_head_b32"]),
        ("CLIP B/32 Full FT", ["full_finetune_b32"]),
    ]:
        for fname in candidates:
            try:
                comparison[display] = load_results(fname)
                break
            except FileNotFoundError:
                continue

    if comparison:
        print_results_table(comparison)
        plot_recall_comparison(comparison, title="All Approaches Comparison")
"""


# Expanded catastrophic-forgetting test: ~30 generic queries spanning multiple categories,
# plus quantitative cosine-similarity drift metric.
CELL_16_FORGETTING = """\
TRAINING_COMPLETE = True  # Set to False to skip forgetting analysis

if TRAINING_COMPLETE:
    # Curated ~30 generic queries spanning categories CLIP saw at scale but
    # Flickr30k under-represents. If FT degrades these, that is forgetting.
    GENERIC_QUERIES = {
        "animals":   ["a photo of a cat", "a black bear in a forest", "an elephant in the savanna",
                      "a colorful parrot on a branch"],
        "vehicles":  ["a red sports car on a highway", "a vintage steam locomotive",
                      "a fighter jet in the sky", "a sailboat at sunset"],
        "food":      ["a plate of sushi", "a slice of pepperoni pizza",
                      "a bowl of ramen with chopsticks", "a chocolate birthday cake"],
        "scenery":   ["snow covered mountains", "sunset over the ocean",
                      "northern lights over a frozen lake", "a desert with sand dunes"],
        "buildings": ["the Eiffel tower at night", "a Gothic cathedral interior",
                      "a futuristic glass skyscraper", "a Japanese pagoda"],
        "art_abstract": ["a Van Gogh style painting", "abstract geometric pattern",
                         "an oil painting of fruit", "a black and white sketch portrait"],
        "tech":      ["a person using a laptop computer", "a smartphone screen showing apps",
                      "a robot arm in a factory", "a circuit board close up"],
        "people":    ["an astronaut on the moon", "a chef cooking in a restaurant"],
    }
    flat_queries = [q for cat_queries in GENERIC_QUERIES.values() for q in cat_queries]

    # Load a fresh zero-shot CLIP for comparison
    zs_model, zs_processor = load_clip_model("clip-vit-b-32", device=device)

    model.eval()
    zs_model.eval()
    with torch.no_grad():
        ft_img_emb = encode_images(splits["test"], model, processor, device=device)
        ft_query_emb = encode_texts(flat_queries, model, processor, device=device)

        zs_img_emb = encode_images(splits["test"], zs_model, zs_processor, device=device)
        zs_query_emb = encode_texts(flat_queries, zs_model, zs_processor, device=device)

    # Quantitative metric: average rank-shift of zero-shot's top-1 image
    # under the fine-tuned model. Large positive shift = forgetting.
    rank_shifts: list[int] = []
    per_query_records: list[dict] = []
    for i, query in enumerate(flat_queries):
        zs_scores = zs_img_emb @ zs_query_emb[i]
        ft_scores = ft_img_emb @ ft_query_emb[i]

        zs_top1_img = int(np.argmax(zs_scores))
        ft_rank_of_zs_top1 = int(np.sum(ft_scores > ft_scores[zs_top1_img]))

        rank_shifts.append(ft_rank_of_zs_top1)
        per_query_records.append({
            "query": query,
            "zs_top1_image_idx": zs_top1_img,
            "ft_rank_of_zs_top1": ft_rank_of_zs_top1,
        })

    rank_shifts_arr = np.array(rank_shifts)
    forgetting_summary = {
        "n_queries": len(flat_queries),
        "mean_rank_shift": float(rank_shifts_arr.mean()),
        "median_rank_shift": float(np.median(rank_shifts_arr)),
        "max_rank_shift": int(rank_shifts_arr.max()),
        "frac_unchanged_top1": float((rank_shifts_arr == 0).mean()),
    }
    print("\\nCatastrophic forgetting summary (30+ generic queries):")
    for k, v in forgetting_summary.items():
        print(f"  {k}: {v}")
    save_results(forgetting_summary, "catastrophic_forgetting_summary")

    # Visualize a handful of comparisons
    from src.retrieval import text_to_image_search
    from src.visualize import plot_retrieval_results_comparison

    rng = np.random.default_rng(42)
    sample_qs = rng.choice(len(flat_queries), size=min(8, len(flat_queries)), replace=False)
    for i in sample_qs:
        ft_results = text_to_image_search(ft_query_emb[i], ft_img_emb, top_k=5)
        zs_results = text_to_image_search(zs_query_emb[i], zs_img_emb, top_k=5)
        fig = plot_retrieval_results_comparison(
            flat_queries[i],
            {"Zero-shot CLIP": zs_results, "Fine-tuned CLIP": ft_results},
            splits["test"],
        )
        fig.show()

    del zs_model, zs_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""


CELL_17_CURVES = """\
TRAINING_COMPLETE = True  # Set to False to skip plot

if TRAINING_COMPLETE:
    plot_training_curves(train_losses, val_metrics_history, title="Full Fine-tuning Training")
    finish_wandb()
"""


def main() -> None:
    nb = load_nb(NB_PATH)
    set_cell_source(nb, 1, CELL_1_IMPORTS)
    set_cell_source(nb, 7, CELL_7_DATA)
    set_cell_source(nb, 9, CELL_9_HYPER)
    set_cell_source(nb, 11, CELL_11_TRAIN)
    set_cell_source(nb, 13, CELL_13_TEST_EVAL)
    set_cell_source(nb, 14, CELL_14_COMPARE)
    set_cell_source(nb, 16, CELL_16_FORGETTING)
    set_cell_source(nb, 17, CELL_17_CURVES)
    save_nb(NB_PATH, nb)
    print(f"Wired {NB_PATH}")


if __name__ == "__main__":
    main()
