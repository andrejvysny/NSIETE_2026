import numpy as np

from src.config import TOP_K_VALUES


def text_to_image_search(
    text_embedding: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Retrieve top-K images for a text query embedding.

    Args:
        text_embedding: (512,) L2-normalized text vector
        image_embeddings: (N, 512) L2-normalized image matrix
        top_k: Number of results

    Returns:
        List of (image_index, similarity_score) tuples, descending by score.
    """
    scores = image_embeddings @ text_embedding  # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def text_to_image_search_batch(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Batch retrieval for multiple text queries.

    Args:
        text_embeddings: (Q, 512) L2-normalized text matrix
        image_embeddings: (N, 512) L2-normalized image matrix
        top_k: Number of results per query

    Returns:
        np.ndarray of shape (Q, top_k) with image indices per query,
        sorted by descending similarity.
    """
    # (Q, 512) @ (512, N) -> (Q, N)
    scores = text_embeddings @ image_embeddings.T
    # argsort descending, take top_k per query
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
    return top_indices


def get_similarity_scores(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
) -> np.ndarray:
    """Full similarity matrix between all texts and images.

    Returns:
        np.ndarray of shape (Q, N) with similarity scores.
    """
    return text_embeddings @ image_embeddings.T


def get_ranks(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    ground_truth_indices: np.ndarray,
) -> np.ndarray:
    """For each text query, find the rank of the ground-truth image.

    Args:
        text_embeddings: (Q, 512)
        image_embeddings: (N, 512)
        ground_truth_indices: (Q,) -- correct image index per query

    Returns:
        np.ndarray of shape (Q,) with 0-based ranks.
    """
    scores = text_embeddings @ image_embeddings.T  # (Q, N)
    # For each query, count how many images have higher similarity than GT
    gt_scores = scores[np.arange(len(ground_truth_indices)), ground_truth_indices]  # (Q,)
    # rank = number of images with strictly higher similarity
    ranks = (scores > gt_scores[:, None]).sum(axis=1)  # (Q,)
    return ranks.astype(np.int64)
