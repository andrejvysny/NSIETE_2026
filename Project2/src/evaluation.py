import json
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR, TOP_K_VALUES
from src.retrieval import get_ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Recall@K: fraction of queries where ground truth is in top-K.

    Args:
        ranks: (Q,) array of 0-based ranks for each query
        k: cutoff value

    Returns:
        Recall@K score in [0, 1]
    """
    return float(np.mean(ranks < k))


def median_rank(ranks: np.ndarray) -> float:
    """Median rank of ground-truth across all queries (1-based)."""
    return float(np.median(ranks + 1))


def mean_rank(ranks: np.ndarray) -> float:
    """Mean rank of ground-truth across all queries (1-based)."""
    return float(np.mean(ranks + 1))


def evaluate_text_to_image(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    ground_truth_indices: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Full text-to-image retrieval evaluation.

    Args:
        text_embeddings: (Q, 512) -- one per caption query
        image_embeddings: (N, 512) -- gallery images
        ground_truth_indices: (Q,) -- correct image index per caption
        k_values: [1, 5, 10] typically

    Returns:
        Dict: {"R@1": 0.xx, "R@5": 0.xx, "R@10": 0.xx,
               "MedianR": xx, "MeanR": xx}
    """
    if k_values is None:
        k_values = TOP_K_VALUES

    ranks = get_ranks(text_embeddings, image_embeddings, ground_truth_indices)

    results = {}
    for k in k_values:
        results[f"R@{k}"] = recall_at_k(ranks, k)
    results["MedianR"] = median_rank(ranks)
    results["MeanR"] = mean_rank(ranks)

    return results


def evaluate_image_to_text(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_indices: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Image-to-text retrieval evaluation (symmetric direction of t2i).

    For each image, find the rank of its best matching caption among all captions.
    """
    if k_values is None:
        k_values = TOP_K_VALUES

    scores = image_embeddings @ text_embeddings.T  # (N, Q)
    n_images = len(image_embeddings)

    ranks_list = []
    for img_idx in range(n_images):
        img_scores = scores[img_idx]
        caption_mask = image_indices == img_idx
        if not caption_mask.any():
            continue
        best_caption_score = img_scores[caption_mask].max()
        rank = int((img_scores > best_caption_score).sum())
        ranks_list.append(rank)

    ranks = np.array(ranks_list, dtype=np.int64)

    results = {}
    for k in k_values:
        results[f"R@{k}"] = recall_at_k(ranks, k)
    results["MedianR"] = median_rank(ranks)
    results["MeanR"] = mean_rank(ranks)
    return results


# ─── Save / Load / Print ───


def save_results(results: dict, name: str) -> Path:
    """Save evaluation results to data/results/{name}.json"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")
    return path


def load_results(name: str) -> dict:
    """Load saved evaluation results."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}")
    with open(path) as f:
        return json.load(f)


def print_results_table(
    results: dict[str, dict[str, float]],
    title: str = "Text-to-Image Retrieval Results",
) -> None:
    """Pretty-print comparison table of multiple model results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    # Collect all metric keys
    all_keys: list[str] = []
    for model_results in results.values():
        for k in model_results:
            if k not in all_keys:
                all_keys.append(k)

    # Header
    header = f"{'Model':<25}"
    for key in all_keys:
        header += f"{key:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for model_name, model_results in results.items():
        row = f"{model_name:<25}"
        for key in all_keys:
            val = model_results.get(key, 0.0)
            if key.startswith("R@"):
                row += f"{val:>9.1%}"
            else:
                row += f"{val:>10.1f}"
        print(row)

    print(f"{'=' * 60}\n")
