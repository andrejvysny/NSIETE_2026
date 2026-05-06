from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import BATCH_SIZE, CLIP_MODELS, EMBEDDINGS_DIR, MODELS_DIR, get_device


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


# ─── Model loading ───


def load_clip_model(
    model_key: str = "clip-vit-b-32",
    device: torch.device | None = None,
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load a CLIP model and processor.

    Args:
        model_key: One of "clip-vit-b-32", "clip-vit-b-16"
        device: Torch device, auto-detected if None

    Returns:
        (model, processor) tuple
    """
    if device is None:
        device = get_device()
    model_name = CLIP_MODELS[model_key]
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, processor


# ─── Fine-tuned model registry ───


class _ProjectionWrappedCLIP(nn.Module):
    """Wraps a base CLIP model + image/text projection heads.

    Exposes ``get_image_features`` and ``get_text_features`` so callers
    (encode_images / encode_texts) work unchanged.
    """

    def __init__(
        self,
        base: CLIPModel,
        image_proj: nn.Module,
        text_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.base = base
        self.image_proj = image_proj
        self.text_proj = text_proj

    def get_image_features(self, **kwargs) -> torch.Tensor:
        feats = self.base.get_image_features(**kwargs)
        return self.image_proj(feats)

    def get_text_features(self, **kwargs) -> torch.Tensor:
        feats = self.base.get_text_features(**kwargs)
        return self.text_proj(feats)


def load_finetuned_model(
    model_name: str,
    device: torch.device | None = None,
) -> tuple[Any, CLIPProcessor, str]:
    """Resolve a model name to (model, processor, kind).

    Recognized names:
        - any key in CLIP_MODELS  → kind="base"
        - "projection_b32"        → frozen B/32 + ProjectionHead pair
        - "lora_b32"              → B/32 + PEFT LoRA adapter
        - "full_b32"              → fully fine-tuned B/32 weights

    Returns:
        (model, processor, kind). The model exposes get_image_features /
        get_text_features regardless of kind.
    """
    if device is None:
        device = get_device()

    if model_name in CLIP_MODELS:
        model, processor = load_clip_model(model_name, device=device)
        return model, processor, "base"

    if model_name == "projection_b32":
        from src.training import ProjectionHead

        base, processor = load_clip_model("clip-vit-b-32", device=device)
        ckpt_path = MODELS_DIR / "projection_head" / "best_projection.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Projection checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        image_proj = ProjectionHead().to(device)
        text_proj = ProjectionHead().to(device)
        image_proj.load_state_dict(ckpt["image_proj_state_dict"])
        text_proj.load_state_dict(ckpt["text_proj_state_dict"])
        wrapped = _ProjectionWrappedCLIP(base, image_proj, text_proj).to(device).eval()
        return wrapped, processor, "projection"

    if model_name == "lora_b32":
        from peft import PeftModel

        base, processor = load_clip_model("clip-vit-b-32", device=device)
        adapter_dir = MODELS_DIR / "lora_clip" / "best"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")
        model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device).eval()
        return model, processor, "lora"

    if model_name == "full_b32":
        base, processor = load_clip_model("clip-vit-b-32", device=device)
        ckpt_path = MODELS_DIR / "full_finetune" / "best_full_finetune.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Full FT checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base.load_state_dict(ckpt["model_state_dict"])
        base.eval()
        return base, processor, "full"

    if model_name.startswith("scratch_vit_") or model_name.startswith("scratch_cnn_"):
        from src.scratch_model import build_cnn_dual, build_vit_dual
        from src.scratch_tokenizer import load_tokenizer

        model_dir = MODELS_DIR / model_name
        ckpt_path = model_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Scratch checkpoint not found: {ckpt_path}")

        tokenizer = load_tokenizer(model_name)
        builder = build_vit_dual if model_name.startswith("scratch_vit_") else build_cnn_dual
        model = builder(
            vocab_size=tokenizer.vocab_size,
            max_length=tokenizer.max_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        model._scratch_tokenizer = tokenizer  # type: ignore[attr-defined]
        kind = "scratch_vit" if model_name.startswith("scratch_vit_") else "scratch_cnn"
        # processor=None is a sentinel: callers must use scratch encoding helpers.
        return model, None, kind

    raise ValueError(f"Unknown model name: {model_name!r}")


# ─── Image encoding ───


def encode_images(
    dataset: Dataset,
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Encode all images in dataset with CLIP image encoder.

    Returns:
        np.ndarray of shape (N, 512), L2-normalized.
    """
    if device is None:
        device = next(model.parameters()).device

    all_embeddings = []
    n = len(dataset)

    for i in tqdm(range(0, n, batch_size), desc="Encoding images"):
        batch = dataset[i : min(i + batch_size, n)]
        images = [_ensure_rgb(img) for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(pixel_values=inputs["pixel_values"])
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 512)
    return _l2_normalize(embeddings)


# ─── Text encoding ───


def encode_texts(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Encode text queries with CLIP text encoder.

    Args:
        texts: List of text strings (captions or queries)

    Returns:
        np.ndarray of shape (len(texts), 512), L2-normalized.
    """
    if device is None:
        device = next(model.parameters()).device

    all_embeddings = []
    n = len(texts)

    for i in tqdm(range(0, n, batch_size), desc="Encoding texts"):
        batch_texts = texts[i : min(i + batch_size, n)]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (Q, 512)
    return _l2_normalize(embeddings)


# ─── Normalization ───


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit norm."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (embeddings / norms).astype(np.float32)


# ─── Save / Load / Cache ───


def save_embeddings(embeddings: np.ndarray, name: str, split: str = "") -> Path:
    """Save embeddings to data/embeddings/{name}[_{split}].npy"""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split}" if split else ""
    path = EMBEDDINGS_DIR / f"{name}{suffix}.npy"
    np.save(path, embeddings)
    print(f"Saved {embeddings.shape} to {path}")
    return path


def load_embeddings(name: str, split: str = "") -> np.ndarray:
    """Load cached embeddings from disk."""
    suffix = f"_{split}" if split else ""
    path = EMBEDDINGS_DIR / f"{name}{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No cached embeddings at {path}")
    return np.load(path)


def extract_and_cache_images(
    dataset: Dataset,
    model_key: str = "clip-vit-b-32",
    split: str = "",
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> np.ndarray:
    """Extract image embeddings with caching.

    Cache key: {model_key}_images[_{split}].npy
    """
    name = f"{model_key}_images"
    suffix = f"_{split}" if split else ""
    path = EMBEDDINGS_DIR / f"{name}{suffix}.npy"

    if path.exists() and not force:
        print(f"Loading cached {name}{suffix} from {path}")
        return np.load(path)

    model, processor = load_clip_model(model_key)
    embeddings = encode_images(dataset, model, processor, batch_size)
    save_embeddings(embeddings, name, split)
    return embeddings


def extract_and_cache_texts(
    texts: list[str],
    model_key: str = "clip-vit-b-32",
    split: str = "",
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> np.ndarray:
    """Extract text embeddings with caching.

    Cache key: {model_key}_texts[_{split}].npy
    """
    name = f"{model_key}_texts"
    suffix = f"_{split}" if split else ""
    path = EMBEDDINGS_DIR / f"{name}{suffix}.npy"

    if path.exists() and not force:
        print(f"Loading cached {name}{suffix} from {path}")
        return np.load(path)

    model, processor = load_clip_model(model_key)
    embeddings = encode_texts(texts, model, processor, batch_size)
    save_embeddings(embeddings, name, split)
    return embeddings
