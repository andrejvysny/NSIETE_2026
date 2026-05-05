"""From-scratch dual-encoder retrieval models.

PyTorch primitives only (no timm, no transformers model classes, no pretrained
weights). Two architectures:

  * ViT image encoder + Transformer text encoder  (build_vit_dual)
  * Mini-ResNet image encoder + Bi-LSTM text encoder  (build_cnn_dual)

Both share a `DualEncoder` joint head that L2-normalizes outputs into a 256-d
shared retrieval space. Output API matches HuggingFace CLIP's
`get_image_features(pixel_values=...)` and `get_text_features(input_ids=,
attention_mask=...)` so that `src.clip_embeddings.encode_*` and
`src.training.training_step` work unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ─── Image preprocessing (no CLIPProcessor, no pretrained stats) ───

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(train: bool) -> transforms.Compose:
    """Standalone resize+normalize pipeline for from-scratch encoders."""
    if train:
        steps = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:
        steps = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    return transforms.Compose(steps)


# ─── Building blocks ───


class PatchEmbed(nn.Module):
    """Conv-based image -> (B, num_patches, dim) embedding."""

    def __init__(self, image_size: int = 224, patch: int = 16, in_chans: int = 3, dim: int = 192) -> None:
        super().__init__()
        if image_size % patch != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch {patch}")
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        self.num_patches = (image_size // patch) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, dim, H/p, W/p) -> (B, num_patches, dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with MultiheadAttention + MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BasicBlock(nn.Module):
    """ResNet-18 style residual block."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


# ─── Image encoders ───


class ImageEncoderViT(nn.Module):
    """Small ViT trained from random init."""

    def __init__(
        self,
        image_size: int = 224,
        patch: int = 16,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch, 3, dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # (B, dim) — CLS token


class ImageEncoderCNN(nn.Module):
    """ResNet-18 layout, trained from random init."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.stem(pixel_values)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x  # (B, 512)


# ─── Text encoders ───


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool (B, T, D) over T using bool/int mask (B, T) where 1 = keep."""
    mask = mask.to(x.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class TextEncoderTransformer(nn.Module):
    """Embedding + sinusoidal/learned positional + transformer blocks + masked mean."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int = 192,
        depth: int = 4,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = input_ids.size(1)
        x = self.token_embed(input_ids) + self.pos_embed[:, :T]
        x = self.dropout(x)
        # nn.MultiheadAttention key_padding_mask uses True = "ignore"
        kpm = None if attention_mask is None else (attention_mask == 0)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        x = self.norm(x)
        if attention_mask is None:
            return x.mean(dim=1)
        return _masked_mean(x, attention_mask)


class TextEncoderLSTM(nn.Module):
    """Embedding + bidirectional LSTM + masked mean over outputs."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden * 2
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(self.token_embed(input_ids))
        out, _ = self.lstm(x)  # (B, T, 2*hidden)
        if attention_mask is None:
            return out.mean(dim=1)
        return _masked_mean(out, attention_mask)


# ─── Dual-tower wrapper ───


class DualEncoder(nn.Module):
    """Image encoder + text encoder + projection heads + L2 normalize."""

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        joint_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(image_encoder.out_dim, joint_dim)
        self.text_proj = nn.Linear(text_encoder.out_dim, joint_dim)
        self.joint_dim = joint_dim
        nn.init.trunc_normal_(self.image_proj.weight, std=0.02)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.trunc_normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)

    def get_image_features(self, pixel_values: torch.Tensor, **_: object) -> torch.Tensor:
        feats = self.image_encoder(pixel_values)
        return F.normalize(self.image_proj(feats), dim=-1)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        feats = self.text_encoder(input_ids, attention_mask=attention_mask)
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get_image_features(pixel_values=pixel_values),
            self.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
        )


# ─── Builders ───


def build_vit_dual(
    vocab_size: int,
    max_length: int,
    pad_token_id: int = 0,
    joint_dim: int = 256,
) -> DualEncoder:
    img = ImageEncoderViT(image_size=224, patch=16, dim=192, depth=6, heads=6)
    txt = TextEncoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        dim=192,
        depth=4,
        heads=6,
        pad_token_id=pad_token_id,
    )
    return DualEncoder(img, txt, joint_dim=joint_dim)


def build_cnn_dual(
    vocab_size: int,
    max_length: int,  # unused for LSTM, kept for API symmetry
    pad_token_id: int = 0,
    joint_dim: int = 256,
) -> DualEncoder:
    del max_length
    img = ImageEncoderCNN()
    txt = TextEncoderLSTM(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden=256,
        num_layers=2,
        pad_token_id=pad_token_id,
    )
    return DualEncoder(img, txt, joint_dim=joint_dim)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
