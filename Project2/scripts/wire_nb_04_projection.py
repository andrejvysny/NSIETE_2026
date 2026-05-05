"""Wire notebook 04 (projection head): seed, wandb, training enabled, embedding cache."""

from __future__ import annotations

from pathlib import Path

from _nb_edit import load_nb, save_nb, set_cell_source

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "04_finetune_projection.ipynb"


CELL_1_IMPORTS = """\
import sys
from pathlib import Path

# ensure project root is on path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gc

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

from src.config import MODELS_DIR, EMBEDDINGS_DIR, get_device, set_seeds
from src.data import load_karpathy_splits, get_all_captions_flat
from src.clip_embeddings import extract_and_cache_images, extract_and_cache_texts
from src.evaluation import evaluate_text_to_image, save_results, load_results, print_results_table
from src.training import (
    ProjectionHead,
    HardNegativeInfoNCELoss,
    save_checkpoint,
    load_checkpoint,
    run_validation_projection,
    init_wandb,
    log_train_step,
    log_val_metrics,
    log_summary,
    log_artifact,
    finish_wandb,
)
from src.visualize import plot_training_curves, plot_recall_comparison, plot_retrieval_results

set_seeds(42)
device = get_device()

# Perf knobs (deterministic=False; we trade strict bit-equality for speed).
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


CELL_12_TRAIN = """\
RUN_TRAINING = True  # Set to False to skip training

if RUN_TRAINING:
    CHECKPOINT_DIR = MODELS_DIR / "projection_head"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    run = init_wandb(
        run_name="projection_head_b32",
        config={
            "approach": "projection_head",
            "model": "clip-vit-b-32",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "temperature": 0.07,
            "hard_negative_weight": 2.0,
            "loss": "HardNegativeInfoNCE",
            "projection_input_dim": 512,
            "projection_hidden_dim": 512,
            "projection_output_dim": 256,
            "seed": 42,
            "device": str(device),
        },
        tags=["projection_head"],
    )

    best_val_r1 = 0.0
    train_losses: list[float] = []
    val_metrics_history: dict[str, list[float]] = {"R@1": [], "R@5": [], "R@10": []}
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        # -- Train --
        image_proj.train()
        text_proj.train()
        epoch_losses: list[float] = []
        ep_start = time.time()

        for text_emb_batch, img_emb_batch in train_loader:
            text_emb_batch = text_emb_batch.to(device, non_blocking=True)
            img_emb_batch = img_emb_batch.to(device, non_blocking=True)

            proj_text = text_proj(text_emb_batch)   # (B, 256)
            proj_img = image_proj(img_emb_batch)    # (B, 256)

            loss = loss_fn(proj_img, proj_text)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            log_train_step(
                {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )
            global_step += 1

        avg_loss = float(np.mean(epoch_losses))

        # -- Validate --
        val_results = run_validation_projection(
            image_proj, text_proj, val_img_emb, val_txt_emb, val_gt, device=device
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
            torch.save(
                {
                    "image_proj_state_dict": image_proj.state_dict(),
                    "text_proj_state_dict": text_proj.state_dict(),
                    "epoch": epoch,
                    "val_results": val_results,
                },
                CHECKPOINT_DIR / "best_projection.pt",
            )
            print(f"  -> New best! Saved checkpoint (R@1={best_val_r1:.3f})")

    print(f"\\nBest validation R@1: {best_val_r1:.3f}")
"""


CELL_14_TEST_EVAL = """\
TRAINING_COMPLETE = True  # Set to False to skip evaluation

if TRAINING_COMPLETE:
    CHECKPOINT_DIR = MODELS_DIR / "projection_head"

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_projection.pt", map_location=device, weights_only=False
    )
    image_proj.load_state_dict(checkpoint["image_proj_state_dict"])
    text_proj.load_state_dict(checkpoint["text_proj_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    image_proj.eval()
    text_proj.eval()
    with torch.no_grad():
        test_img_proj = image_proj(torch.from_numpy(test_img_emb).float().to(device)).cpu().numpy()
        test_txt_proj = text_proj(torch.from_numpy(test_txt_emb).float().to(device)).cpu().numpy()

    # Cache projected test embeddings under the app's discovery pattern
    np.save(EMBEDDINGS_DIR / "projection_b32_images_test.npy", test_img_proj)
    np.save(EMBEDDINGS_DIR / "projection_b32_texts_test.npy", test_txt_proj)
    print(
        f"Cached projected test embeddings: "
        f"{test_img_proj.shape} images, {test_txt_proj.shape} texts"
    )

    proj_results = evaluate_text_to_image(test_txt_proj, test_img_proj, test_gt)
    save_results(proj_results, "projection_head_b32")
    print("\\nProjection Head Results:")
    for k, v in proj_results.items():
        print(f"  {k}: {v:.1%}" if k.startswith("R@") else f"  {k}: {v:.4f}")

    # Log final test metrics + checkpoint artifact to wandb (if a run is active)
    log_summary(proj_results, prefix="test")
    log_artifact(
        CHECKPOINT_DIR / "best_projection.pt",
        name="projection_head_b32",
        artifact_type="model",
    )
    finish_wandb()
"""


CELL_15_COMPARE = """\
TRAINING_COMPLETE = True  # Set to False to skip comparison

if TRAINING_COMPLETE:
    try:
        baseline_results = load_results("baseline_b32")
    except FileNotFoundError:
        try:
            baseline_results = load_results("baseline_clip-vit-b-32")
        except FileNotFoundError:
            baseline_results = evaluate_text_to_image(test_txt_emb, test_img_emb, test_gt)
            save_results(baseline_results, "baseline_clip-vit-b-32")

    comparison = {
        "CLIP B/32 (baseline)": baseline_results,
        "CLIP B/32 + Projection": proj_results,
    }
    print_results_table(comparison)
    plot_recall_comparison(comparison, title="Baseline vs Projection Head")
"""


CELL_17_CURVES = """\
TRAINING_COMPLETE = True  # Set to False to skip plot

if TRAINING_COMPLETE:
    plot_training_curves(train_losses, val_metrics_history, title="Projection Head Training")
"""


CELL_18_QUAL = """\
TRAINING_COMPLETE = True  # Set to False to skip qualitative examples

if TRAINING_COMPLETE:
    from src.retrieval import text_to_image_search

    n_examples = 5
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(test_captions), size=n_examples, replace=False)

    for idx in sample_indices:
        query = test_captions[idx]
        gt_img_idx = int(test_gt[idx])

        results = text_to_image_search(test_txt_proj[idx], test_img_proj, top_k=5)
        fig = plot_retrieval_results(
            query, results, splits["test"], ground_truth_idx=gt_img_idx
        )
        fig.show()
"""


def main() -> None:
    nb = load_nb(NB_PATH)
    set_cell_source(nb, 1, CELL_1_IMPORTS)
    set_cell_source(nb, 12, CELL_12_TRAIN)
    set_cell_source(nb, 14, CELL_14_TEST_EVAL)
    set_cell_source(nb, 15, CELL_15_COMPARE)
    set_cell_source(nb, 17, CELL_17_CURVES)
    set_cell_source(nb, 18, CELL_18_QUAL)
    save_nb(NB_PATH, nb)
    print(f"Wired {NB_PATH}")


if __name__ == "__main__":
    main()
