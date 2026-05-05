import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.manifold import TSNE


# ─── Text-to-Image Retrieval Visualization ───


def plot_retrieval_results(
    query_text: str,
    results: list[tuple[int, float]],
    dataset: Dataset,
    ground_truth_idx: int | None = None,
    title: str = "",
) -> plt.Figure:
    """Display text query + top-K retrieved images.

    Green border = correct (ground truth), red border = incorrect.
    """
    k = len(results)
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 4))
    if k == 1:
        axes = [axes]

    for i, (idx, score) in enumerate(results):
        axes[i].imshow(dataset[idx]["image"])
        label = f"#{i+1} sim={score:.3f}"

        if ground_truth_idx is not None:
            is_correct = idx == ground_truth_idx
            color = "green" if is_correct else "red"
            for spine in axes[i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            if is_correct:
                label += " ✓"
        axes[i].set_title(label, fontsize=9)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Wrap query text
    wrapped = query_text[:80] + ("..." if len(query_text) > 80 else "")
    suptitle = title or f'Query: "{wrapped}"'
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_retrieval_results_comparison(
    query_text: str,
    results_by_model: dict[str, list[tuple[int, float]]],
    dataset: Dataset,
    ground_truth_idx: int | None = None,
) -> plt.Figure:
    """Side-by-side retrieval results from multiple models."""
    n_models = len(results_by_model)
    model_names = list(results_by_model.keys())
    k = max(len(r) for r in results_by_model.values())

    fig, axes = plt.subplots(n_models, k, figsize=(3 * k, 3.5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row, model_name in enumerate(model_names):
        results = results_by_model[model_name]
        for col in range(k):
            ax = axes[row][col] if k > 1 else axes[row]
            if col < len(results):
                idx, score = results[col]
                ax.imshow(dataset[idx]["image"])
                label = f"sim={score:.3f}"
                if ground_truth_idx is not None and idx == ground_truth_idx:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("green")
                        spine.set_linewidth(3)
                    label += " ✓"
                ax.set_title(label, fontsize=8)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(model_name, fontsize=10, fontweight="bold")

    wrapped = query_text[:80] + ("..." if len(query_text) > 80 else "")
    fig.suptitle(f'Query: "{wrapped}"', fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── Metric Visualization ───


def plot_recall_comparison(
    results: dict[str, dict[str, float]],
    title: str = "Recall@K Comparison",
) -> plt.Figure:
    """Bar chart comparing R@1, R@5, R@10 across models."""
    models = list(results.keys())
    recall_keys = sorted(
        [k for k in results[models[0]] if k.startswith("R@")],
        key=lambda x: int(x.split("@")[1]),
    )

    x = np.arange(len(recall_keys))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        values = [results[model].get(k, 0) * 100 for k in recall_keys]
        bars = ax.bar(x + i * width, values, width, label=model)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Recall (%)")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(recall_keys)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_rank_distribution(
    ranks: np.ndarray,
    title: str = "Rank Distribution",
    max_rank: int = 100,
) -> plt.Figure:
    """Histogram of ground-truth ranks, clipped to max_rank."""
    clipped = np.minimum(ranks + 1, max_rank)  # 1-based

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clipped, bins=min(max_rank, 50), alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Rank of Ground Truth Image")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Annotate key thresholds
    for k, color in [(1, "green"), (5, "orange"), (10, "red")]:
        frac = (ranks < k).mean() * 100
        ax.axvline(k + 0.5, color=color, linestyle="--",
                   label=f"R@{k} = {frac:.1f}%")

    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─── Similarity Distribution ───


def plot_text_image_similarity_distribution(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    ground_truth_indices: np.ndarray,
    title: str = "Text-Image Similarity Distribution",
    n_negatives: int = 50000,
) -> plt.Figure:
    """Separate histograms for positive pairs vs negative pairs."""
    # Positive pair similarities
    pos_sims = np.sum(
        text_embeddings * image_embeddings[ground_truth_indices], axis=1
    )

    # Random negative pair similarities
    rng = np.random.default_rng(42)
    n_queries = len(text_embeddings)
    rand_text_idx = rng.integers(0, n_queries, size=n_negatives)
    rand_img_idx = rng.integers(0, len(image_embeddings), size=n_negatives)
    neg_sims = np.sum(
        text_embeddings[rand_text_idx] * image_embeddings[rand_img_idx], axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_sims, bins=100, alpha=0.6, label="Negative pairs", color="red", density=True)
    ax.hist(pos_sims, bins=50, alpha=0.7, label="Positive pairs", color="green", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_similarity_distribution(
    embeddings: np.ndarray,
    title: str = "Pairwise Similarity Distribution",
    sample_pairs: int = 50000,
) -> plt.Figure:
    """Histogram of cosine similarities for random pairs."""
    n = len(embeddings)
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=sample_pairs)
    idx_b = rng.integers(0, n, size=sample_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=100, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(np.mean(sims), color="red", linestyle="--", label=f"mean={np.mean(sims):.3f}")
    ax.legend()
    fig.tight_layout()
    return fig


# ─── Embedding Space ───


def plot_tsne(
    embeddings: np.ndarray,
    dataset: Dataset,
    title: str = "t-SNE",
    sample_size: int = 2000,
    labels: list[str] | None = None,
    perplexity: int = 30,
) -> plt.Figure:
    """2D t-SNE scatter plot of embeddings with caption-derived labels."""
    n = len(embeddings)
    rng = np.random.default_rng(42)

    if sample_size < n:
        indices = rng.choice(n, size=sample_size, replace=False)
    else:
        indices = np.arange(n)

    subset = embeddings[indices]

    print(f"Running t-SNE on {len(indices)} points...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(subset)

    if labels is None:
        keywords = ["dog", "man", "woman", "child", "water", "sport", "car", "food"]
        labels_arr = []
        for idx in indices:
            captions = " ".join(dataset[int(idx)]["caption"]).lower()
            found = "other"
            for kw in keywords:
                if kw in captions:
                    found = kw
                    break
            labels_arr.append(found)
    else:
        labels_arr = [labels[i] for i in indices]

    unique_labels = sorted(set(labels_arr))
    color_map = {l: i for i, l in enumerate(unique_labels)}
    colors = [color_map[l] for l in labels_arr]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap="tab10", alpha=0.5, s=8,
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=plt.cm.tab10(color_map[l] / max(len(unique_labels), 1)),
            markersize=8, label=l,
        )
        for l in unique_labels
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ─── Training Curves ───


def plot_training_curves(
    train_losses: list[float],
    val_metrics: dict[str, list[float]] | None = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Training loss curve + optional validation Recall@K over epochs."""
    n_subplots = 1 + (1 if val_metrics else 0)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(train_losses, "b-", linewidth=1.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation metrics
    if val_metrics and n_subplots > 1:
        for metric_name, values in val_metrics.items():
            label = metric_name
            if metric_name.startswith("R@"):
                values_pct = [v * 100 for v in values]
                axes[1].plot(values_pct, "-o", label=label, markersize=4)
            else:
                axes[1].plot(values, "-o", label=label, markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation Metrics")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── Error Analysis ───


def plot_error_analysis_grid(
    failure_cases: list[dict],
    dataset: Dataset,
    n_cols: int = 4,
    n_rows: int = 3,
) -> plt.Figure:
    """Grid of failure cases.

    Each failure_case dict has:
        - "query_text": str
        - "gt_idx": int (ground truth image index)
        - "retrieved_idx": int (top-1 retrieved image index)
        - "gt_rank": int (rank of ground truth)
    """
    n_show = min(len(failure_cases), n_cols * n_rows)
    fig, axes = plt.subplots(n_show, 2, figsize=(8, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        case = failure_cases[i]
        gt_idx = case["gt_idx"]
        retrieved_idx = case["retrieved_idx"]

        # Ground truth
        axes[i][0].imshow(dataset[gt_idx]["image"])
        axes[i][0].set_title("Ground Truth", fontsize=9, color="green")
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])

        # Top-1 retrieved (wrong)
        axes[i][1].imshow(dataset[retrieved_idx]["image"])
        axes[i][1].set_title(f"Retrieved (rank={case.get('gt_rank', '?')})", fontsize=9, color="red")
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])

        # Query text as ylabel
        query = case["query_text"][:50] + ("..." if len(case["query_text"]) > 50 else "")
        axes[i][0].set_ylabel(query, fontsize=7, rotation=0, labelpad=120, va="center")

    fig.suptitle("Error Analysis: GT vs Retrieved", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
