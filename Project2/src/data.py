import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from PIL import Image

from src.config import (
    DATA_DIR,
    DATASET_DIR,
    HF_DATASET,
    KARPATHY_JSON_FILENAME,
    KARPATHY_SPLIT_URL,
    PROJECT_ROOT,
)

load_dotenv(PROJECT_ROOT / ".env")


# ─── Dataset download & loading ───


def download_flickr30k(out_dir: Path = DATASET_DIR) -> None:
    """Download Flickr30k from HuggingFace and save to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, token=token)
    ds.save_to_disk(str(out_dir))
    print(f"Saved to {out_dir}")


def load_flickr30k_full(data_dir: Path = DATASET_DIR) -> Dataset:
    """Load all 31,783 Flickr30k images as a single flat Dataset.

    Downloads if not present. Returns Dataset with columns:
    image, caption, sentids, img_id, filename.
    """
    if not (data_dir / "dataset_dict.json").exists() and not (data_dir / "state.json").exists():
        download_flickr30k(data_dir)

    ds = load_from_disk(str(data_dir))

    if isinstance(ds, DatasetDict):
        all_splits = [ds[split] for split in ds]
        return concatenate_datasets(all_splits)
    return ds


# ─── Karpathy splits ───


def download_karpathy_splits(out_dir: Path = DATA_DIR) -> Path:
    """Download Karpathy's dataset_flickr30k.json if not present.

    Downloads the zip from Stanford, extracts the JSON.
    Returns path to the JSON file.
    """
    json_path = out_dir / KARPATHY_JSON_FILENAME
    if json_path.exists():
        return json_path

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Karpathy splits from {KARPATHY_SPLIT_URL}...")
    response = urllib.request.urlopen(KARPATHY_SPLIT_URL)
    zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # Find the JSON inside the zip
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_names:
            raise FileNotFoundError("No JSON found in Karpathy zip")
        with zf.open(json_names[0]) as f:
            data = json.load(f)
        with open(json_path, "w") as out_f:
            json.dump(data, out_f)

    print(f"Saved Karpathy splits to {json_path}")
    return json_path


def _build_filename_to_index(dataset: Dataset) -> dict[str, int]:
    """Build filename -> dataset index mapping."""
    mapping = {}
    for i in range(len(dataset)):
        mapping[dataset[i]["filename"]] = i
    return mapping


def load_karpathy_splits(
    data_dir: Path = DATA_DIR,
    dataset_dir: Path = DATASET_DIR,
) -> dict[str, Dataset]:
    """Load Flickr30k with Karpathy train/val/test splits.

    Returns dict with keys "train", "val", "test", each an HF Dataset.
    Split sizes: ~29,000 train / 1,000 val / 1,000 test.
    "restval" is merged into "train" (standard convention).
    """
    full_ds = load_flickr30k_full(dataset_dir)

    json_path = data_dir / KARPATHY_JSON_FILENAME
    if not json_path.exists():
        json_path = download_karpathy_splits(data_dir)

    filename_to_idx = _build_filename_to_index(full_ds)

    with open(json_path) as f:
        karpathy = json.load(f)

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    missing = 0

    for entry in karpathy["images"]:
        fname = entry["filename"]
        split = entry["split"]
        if split == "restval":
            split = "train"

        if fname in filename_to_idx:
            split_indices[split].append(filename_to_idx[fname])
        else:
            missing += 1

    if missing > 0:
        print(f"Warning: {missing} Karpathy filenames not found in HF dataset")

    splits = {}
    for split_name, indices in split_indices.items():
        indices.sort()
        splits[split_name] = full_ds.select(indices)
        print(f"  {split_name}: {len(indices)} images")

    return splits


# ─── Caption utilities ───


def get_image_caption_pairs(dataset: Dataset) -> list[tuple[int, str]]:
    """Flatten dataset into (image_index, caption) pairs.

    Each image has 5 captions -> 5 pairs per image.
    Returns list of (dataset_index, caption_text).
    """
    pairs = []
    for i in range(len(dataset)):
        for cap in dataset[i]["caption"]:
            pairs.append((i, cap))
    return pairs


def get_all_captions_flat(dataset: Dataset) -> tuple[list[str], np.ndarray]:
    """Get all captions flat with ground truth image indices.

    Returns:
        captions: list of all caption strings (N*5)
        gt_indices: np.ndarray of shape (N*5,) mapping each caption
                    to its image index in the dataset
    """
    captions = []
    gt_indices = []
    for i in range(len(dataset)):
        for cap in dataset[i]["caption"]:
            captions.append(cap)
            gt_indices.append(i)
    return captions, np.array(gt_indices, dtype=np.int64)


def clean_captions(captions: list[str]) -> list[str]:
    """Basic caption cleaning: strip whitespace, remove empty."""
    cleaned = []
    for cap in captions:
        cap = cap.strip()
        if cap:
            cleaned.append(cap)
    return cleaned


# ─── Helpers ───


def get_image(dataset: Dataset, idx: int) -> Image.Image:
    """Get PIL Image at index."""
    return dataset[idx]["image"]


def get_captions(dataset: Dataset, idx: int) -> list[str]:
    """Get list of 5 captions at index."""
    return dataset[idx]["caption"]


def get_filename(dataset: Dataset, idx: int) -> str:
    """Get image filename at index."""
    return dataset[idx]["filename"]


if __name__ == "__main__":
    splits = load_karpathy_splits()
    for name, ds in splits.items():
        print(f"{name}: {len(ds)} images")
    print(f"Test sample captions: {get_captions(splits['test'], 0)}")
