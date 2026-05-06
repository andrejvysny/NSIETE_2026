import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "flickr30k"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# CLIP model identifiers
CLIP_MODELS = {
    "clip-vit-b-32": "openai/clip-vit-base-patch32",
    "clip-vit-b-16": "openai/clip-vit-base-patch16",
}
DEFAULT_CLIP_MODEL = "clip-vit-b-32"

# Karpathy splits
KARPATHY_SPLIT_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip"
KARPATHY_JSON_FILENAME = "dataset_flickr30k.json"

# Retrieval
TOP_K_VALUES = [1, 5, 10]
BATCH_SIZE = 64
IMAGE_SIZE = 224

# Training defaults
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
TEMPERATURE = 0.07  # InfoNCE temperature

# Dataset
HF_DATASET = "lmms-lab/flickr30k"

# Reproducibility
DEFAULT_SEED = 42

# Experiment tracking
WANDB_PROJECT = "nsiete-flickr30k-clip"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA/MPS) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
