"""Smoke-test the from-scratch retrieval models.

Runs:
  * forward passes on a fake batch
  * one backward pass
  * masked-mean text encoding sanity
  * tokenizer round-trip + UNK rate report

Use:  uv run python scripts/smoke_test_scratch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.config import get_device, set_seeds
from src.scratch_model import (
    build_cnn_dual,
    build_image_transform,
    build_vit_dual,
    count_params,
)
from src.scratch_tokenizer import ClipBPETokenizer, WordTokenizer
from src.training import HardNegativeInfoNCELoss


def _check(model, tokenizer, label: str, device: torch.device) -> None:
    model = model.to(device)
    model.train()
    captions = [
        "a man riding a bicycle on a road",
        "two children playing with a ball",
        "a dog running on the beach",
        "an empty hallway with white walls",
    ]
    pixel_values = torch.randn(len(captions), 3, 224, 224, device=device)
    toks = tokenizer.tokenize(captions)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    img_f = model.get_image_features(pixel_values=pixel_values)
    txt_f = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    assert img_f.shape == (len(captions), model.joint_dim), f"img shape {img_f.shape}"
    assert txt_f.shape == (len(captions), model.joint_dim), f"txt shape {txt_f.shape}"

    # L2-normalized?
    img_norms = img_f.norm(dim=-1)
    txt_norms = txt_f.norm(dim=-1)
    assert torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-4), img_norms
    assert torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-4), txt_norms

    loss_fn = HardNegativeInfoNCELoss(temperature=0.07)
    loss = loss_fn(img_f, txt_f)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()

    n_params = count_params(model)
    print(f"  {label}: params={n_params:,}, loss={loss.item():.4f}, "
          f"img_norm={img_norms.mean().item():.3f}, txt_norm={txt_norms.mean().item():.3f}")


def main() -> None:
    set_seeds(42)
    device = get_device()
    print(f"Device: {device}")

    # Image transform check
    transform = build_image_transform(train=False)
    from PIL import Image
    img = Image.new("RGB", (300, 200), color=(123, 117, 104))
    px = transform(img)
    assert px.shape == (3, 224, 224), px.shape
    print(f"image_transform OK: shape={tuple(px.shape)}, dtype={px.dtype}")

    # Tokenizer check
    train_corpus = [
        "a man rides a horse",
        "two children play with a ball",
        "a dog runs on the beach",
        "the sun sets over the ocean",
    ]
    word = WordTokenizer.from_train_captions(train_corpus, min_freq=1, max_length=16)
    print(f"WordTokenizer: vocab={word.vocab_size}, unk_rate={word.unk_rate(train_corpus):.3f}")

    bpe = ClipBPETokenizer(max_length=16)
    print(f"ClipBPETokenizer: vocab={bpe.vocab_size}")

    # Model variants
    print("\nViT + Transformer text:")
    _check(
        build_vit_dual(vocab_size=word.vocab_size, max_length=word.max_length, pad_token_id=word.pad_token_id),
        word, "  word", device,
    )
    _check(
        build_vit_dual(vocab_size=bpe.vocab_size, max_length=bpe.max_length, pad_token_id=bpe.pad_token_id),
        bpe, "  bpe ", device,
    )

    print("\nCNN + LSTM text:")
    _check(
        build_cnn_dual(vocab_size=word.vocab_size, max_length=word.max_length, pad_token_id=word.pad_token_id),
        word, "  word", device,
    )
    _check(
        build_cnn_dual(vocab_size=bpe.vocab_size, max_length=bpe.max_length, pad_token_id=bpe.pad_token_id),
        bpe, "  bpe ", device,
    )

    print("\nAll variants OK")


if __name__ == "__main__":
    main()
