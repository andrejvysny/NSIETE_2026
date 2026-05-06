from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import CLIPModel, CLIPProcessor

from src.config import BATCH_SIZE, TEMPERATURE, get_device

# ─── Datasets ───


class CLIPRetrievalDataset(TorchDataset):
    """Dataset yielding (image, caption_text) pairs for CLIP fine-tuning.

    Each image has 5 captions. Randomly samples one caption per image
    per __getitem__ call. Returns pre-processed pixel values and raw text.
    """

    def __init__(self, hf_dataset: Dataset, processor: CLIPProcessor) -> None:
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        captions = item["caption"]
        caption = captions[np.random.randint(len(captions))]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        return {"pixel_values": pixel_values, "caption": caption}


def make_collate_fn(processor: CLIPProcessor) -> Callable[[list[dict]], dict]:
    """Create a collate function bound to a specific processor.

    Stacks pixel_values, tokenizes captions in batch (max_length=77 for CLIP).
    """

    def _collate(batch: list[dict]) -> dict:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        captions = [item["caption"] for item in batch]
        text_inputs = processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    return _collate


# ─── From-scratch dataset (no CLIPProcessor, custom tokenizer) ───


class ScratchRetrievalDataset(TorchDataset):
    """Yields (image_tensor, caption_text) pairs for from-scratch encoders.

    Image preprocessing happens via a torchvision transform (no CLIPProcessor).
    Tokenization happens later in the collate fn so it can batch-tokenize.
    """

    def __init__(self, hf_dataset: Dataset, image_transform) -> None:
        self.dataset = hf_dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        captions = item["caption"]
        caption = captions[np.random.randint(len(captions))]
        pixel_values = self.image_transform(image)
        return {"pixel_values": pixel_values, "caption": caption}


def make_scratch_collate_fn(tokenizer) -> Callable[[list[dict]], dict]:
    """Collate function bound to a from-scratch tokenizer (Word or BPE)."""

    def _collate(batch: list[dict]) -> dict:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        captions = [item["caption"] for item in batch]
        toks = tokenizer.tokenize(captions)
        return {
            "pixel_values": pixel_values,
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

    return _collate


# ─── Loss Functions ───


class HardNegativeInfoNCELoss(nn.Module):
    """InfoNCE with in-batch hard negative weighting.

    For each positive pair, emphasizes the hardest negatives in the batch by
    scaling their logits with `logits *= (1 + w * softmax(neg_logits))`.
    """

    def __init__(
        self,
        temperature: float = TEMPERATURE,
        hard_negative_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        image_embeddings: torch.Tensor,  # (B, D)
        text_embeddings: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # (B, B) similarity matrix
        logits = (image_embeddings @ text_embeddings.T) / self.temperature
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        # Create mask for positive pairs (diagonal)
        pos_mask = torch.eye(B, device=logits.device, dtype=torch.bool)

        # Weight hard negatives: scale up non-diagonal entries
        # that have high similarity (hard negatives)
        with torch.no_grad():
            # For image-to-text: hardest text negative per image
            neg_logits_i2t = logits.clone()
            neg_logits_i2t[pos_mask] = float("-inf")
            hard_neg_weight_i2t = torch.softmax(neg_logits_i2t, dim=1)

            # For text-to-image: hardest image negative per text
            neg_logits_t2i = logits.T.clone()
            neg_logits_t2i[pos_mask] = float("-inf")
            hard_neg_weight_t2i = torch.softmax(neg_logits_t2i, dim=1)

        # Apply hard negative weighting to logits
        weighted_logits_i2t = logits.clone()
        weighted_logits_i2t[~pos_mask] *= (
            1.0 + self.hard_negative_weight * hard_neg_weight_i2t[~pos_mask]
        )

        weighted_logits_t2i = logits.T.clone()
        weighted_logits_t2i[~pos_mask] *= (
            1.0 + self.hard_negative_weight * hard_neg_weight_t2i[~pos_mask]
        )

        loss_i2t = F.cross_entropy(weighted_logits_i2t, labels)
        loss_t2i = F.cross_entropy(weighted_logits_t2i, labels)
        return (loss_i2t + loss_t2i) / 2


def mine_hard_negatives_offline(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    n_negatives: int = 5,
) -> np.ndarray:
    """For each (text, gt_image) pair return the top-n highest-similarity images
    that aren't the ground truth. Used by the qualitative hard-negative figure
    in the CLIP fine-tuning notebook.
    """
    Q = len(text_embeddings)
    hard_negatives = np.zeros((Q, n_negatives), dtype=np.int64)
    batch_size = 256
    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        scores = text_embeddings[start:end] @ image_embeddings.T
        for i, gt_idx in enumerate(ground_truth[start:end]):
            scores[i, gt_idx] = -np.inf
        hard_negatives[start:end] = np.argsort(scores, axis=1)[:, ::-1][:, :n_negatives]
    return hard_negatives


# ─── Projection Head ───


class ProjectionHead(nn.Module):
    """Learnable projection head on top of frozen CLIP features.

    Maps CLIP's 512-dim joint space to a new space optimized
    for the target dataset.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ─── Training Utilities ───


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """AdamW optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup + cosine decay scheduler."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def training_step(
    model: CLIPModel,
    batch: dict,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single training step for CLIP fine-tuning.

    Returns (loss, metrics_dict).
    """
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    image_features = model.get_image_features(pixel_values=pixel_values)
    text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    # peft.PeftModel returns a BaseModelOutputWithPooling here instead of a
    # raw tensor (see transformers ModelOutput). Unwrap consistently with
    # encode_images / encode_texts in src.clip_embeddings.
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    loss = loss_fn(image_features, text_features)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss encountered: {loss.item()}. "
            "Check for exploding logits, vanishing embeddings, or fp16 underflow."
        )

    with torch.no_grad():
        # Compute in-batch retrieval accuracy for monitoring
        image_norm = F.normalize(image_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        sim = image_norm @ text_norm.T
        acc_i2t = (sim.argmax(dim=1) == torch.arange(len(sim), device=device)).float().mean()
        acc_t2i = (sim.argmax(dim=0) == torch.arange(len(sim), device=device)).float().mean()

    metrics = {
        "loss": loss.item(),
        "acc_i2t": acc_i2t.item(),
        "acc_t2i": acc_t2i.item(),
    }
    return loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load training checkpoint. Returns checkpoint dict."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ─── Validation helpers ───


def run_validation(
    model: nn.Module,
    processor: CLIPProcessor,
    val_dataset: Dataset,
    val_captions: list[str],
    val_gt: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run text-to-image retrieval validation for full/LoRA fine-tuning.

    Encodes val images and captions with the current model state, then
    computes Recall@K, MedianR, MeanR via ``evaluate_text_to_image``.

    Args:
        model: a CLIPModel or PeftModel exposing get_image_features / get_text_features
        val_dataset: HF dataset for images
        val_captions: flat list of captions
        val_gt: (Q,) ground-truth image index per caption
    """
    from src.clip_embeddings import encode_images, encode_texts
    from src.evaluation import evaluate_text_to_image

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            img_emb = encode_images(
                val_dataset, model, processor, batch_size=batch_size, device=device
            )
            txt_emb = encode_texts(
                val_captions, model, processor, batch_size=batch_size, device=device
            )
        return evaluate_text_to_image(txt_emb, img_emb, val_gt)
    finally:
        if was_training:
            model.train()


def run_validation_projection(
    image_proj: nn.Module,
    text_proj: nn.Module,
    val_img_emb: np.ndarray,
    val_txt_emb: np.ndarray,
    val_gt: np.ndarray,
    device: torch.device,
) -> dict:
    """Validation for the projection-head approach.

    Projects pre-computed CLIP embeddings, then evaluates retrieval.
    """
    from src.evaluation import evaluate_text_to_image

    image_proj.eval()
    text_proj.eval()
    with torch.no_grad():
        img_t = torch.from_numpy(val_img_emb).to(device)
        txt_t = torch.from_numpy(val_txt_emb).to(device)
        img_p = image_proj(img_t).cpu().numpy()
        txt_p = text_proj(txt_t).cpu().numpy()
    return evaluate_text_to_image(txt_p, img_p, val_gt)


# ─── From-scratch encoding + validation ───


def encode_images_scratch(
    hf_dataset,
    model: nn.Module,
    image_transform,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode images for from-scratch DualEncoder using a torchvision transform."""
    model.eval()
    n = len(hf_dataset)
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = hf_dataset[start:end]
            imgs = []
            for img in batch["image"]:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                imgs.append(image_transform(img))
            pixel_values = torch.stack(imgs).to(device)
            out = model.get_image_features(pixel_values=pixel_values)
            feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def encode_texts_scratch(
    captions: list[str],
    model: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode captions for from-scratch DualEncoder using a custom tokenizer."""
    model.eval()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            end = min(start + batch_size, len(captions))
            tok = tokenizer.tokenize(captions[start:end])
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            out = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def run_validation_scratch(
    model: nn.Module,
    tokenizer,
    image_transform,
    val_dataset,
    val_captions: list[str],
    val_gt: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Validation for from-scratch DualEncoder.

    Re-encodes val images + captions with the current model state, then computes
    Recall@K via ``evaluate_text_to_image``. Always runs in eval mode regardless
    of caller state and restores afterwards.
    """
    from src.evaluation import evaluate_text_to_image

    was_training = model.training
    model.eval()
    try:
        img_emb = encode_images_scratch(
            val_dataset, model, image_transform, device=device, batch_size=batch_size
        )
        txt_emb = encode_texts_scratch(
            val_captions, model, tokenizer, device=device, batch_size=batch_size * 2
        )
        return evaluate_text_to_image(txt_emb, img_emb, val_gt)
    finally:
        if was_training:
            model.train()


# ─── WandB helpers ───
#
# Lightweight wrappers so notebooks don't import wandb directly. All helpers
# no-op when wandb is not initialized, so the same training code runs with or
# without tracking enabled.


def init_wandb(
    run_name: str,
    config: dict,
    project: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
) -> Any:
    """Initialize a wandb run. mode='disabled' → silent no-op. Returns the wandb Run."""
    import wandb

    from src.config import WANDB_PROJECT

    return wandb.init(
        project=project or WANDB_PROJECT,
        name=run_name,
        config=config,
        tags=tags or [],
        mode=mode,
        reinit=True,
    )


def log_train_step(metrics: dict, step: int) -> None:
    """Log per-step training metrics under the train/* namespace."""
    import wandb

    if wandb.run is not None:
        wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)


def log_val_metrics(metrics: dict, epoch: int) -> None:
    """Log per-epoch validation metrics under the val/* namespace."""
    import wandb

    if wandb.run is not None:
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)


def log_summary(metrics: dict, prefix: str = "test") -> None:
    """Add final test metrics to wandb run summary (sticky, not time-series)."""
    import wandb

    if wandb.run is not None:
        wandb.summary.update({f"{prefix}/{k}": v for k, v in metrics.items()})


def log_artifact(path: Path, name: str, artifact_type: str = "model") -> None:
    """Upload a small file as a wandb Artifact (skip for >100 MB checkpoints)."""
    import wandb

    if wandb.run is None:
        return
    art = wandb.Artifact(name, type=artifact_type)
    art.add_file(str(path))
    wandb.log_artifact(art)


def finish_wandb() -> None:
    """Close the current wandb run."""
    import wandb

    if wandb.run is not None:
        wandb.finish()
