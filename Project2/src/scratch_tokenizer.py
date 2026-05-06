"""From-scratch tokenizers for retrieval encoders.

Two implementations sharing a small interface:

  * WordTokenizer       -- word-level vocab built from train captions
  * ClipBPETokenizer    -- thin wrapper around the CLIP BPE tokenizer
                           (ONLY the tokenization vocab; no pretrained
                           embeddings -- the from-scratch text encoder
                           trains its own nn.Embedding from random init)

Both expose:
    .tokenize(texts: list[str]) -> {"input_ids": LongTensor (B, T),
                                    "attention_mask": LongTensor (B, T)}
    .vocab_size: int
    .max_length: int
    .pad_token_id: int
    .save(dir: Path)  / .load(dir: Path)  (class-method)
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import torch

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
CLS_ID = 1
UNK_ID = 2
EOS_ID = 3


# ─── Word-level tokenizer ───


_WORD_RE = re.compile(r"[a-z']+")


def _split_caption(caption: str) -> list[str]:
    return _WORD_RE.findall(caption.lower())


class WordTokenizer:
    """Tokenizer with a small word-level vocabulary built from training captions."""

    name = "word"

    def __init__(
        self,
        word2id: dict[str, int],
        max_length: int = 32,
    ) -> None:
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.max_length = max_length
        self.pad_token_id = PAD_ID
        self.cls_token_id = CLS_ID
        self.unk_token_id = UNK_ID
        self.eos_token_id = EOS_ID

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    @classmethod
    def from_train_captions(
        cls,
        captions: Iterable[str],
        min_freq: int = 3,
        max_vocab: int = 20000,
        max_length: int = 32,
    ) -> WordTokenizer:
        from collections import Counter

        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(_split_caption(cap))

        word2id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            CLS_TOKEN: CLS_ID,
            UNK_TOKEN: UNK_ID,
            EOS_TOKEN: EOS_ID,
        }
        capacity = max(0, max_vocab - len(word2id))
        ranked = [w for w, c in counter.most_common() if c >= min_freq]
        for w in ranked[:capacity]:
            word2id[w] = len(word2id)
        return cls(word2id, max_length=max_length)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        all_ids: list[list[int]] = []
        for text in texts:
            toks = _split_caption(text)[: self.max_length - 2]
            ids = [CLS_ID] + [self.word2id.get(t, UNK_ID) for t in toks] + [EOS_ID]
            ids = ids[: self.max_length]
            ids += [PAD_ID] * (self.max_length - len(ids))
            all_ids.append(ids)
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = (input_ids != PAD_ID).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def unk_rate(self, captions: Iterable[str]) -> float:
        total = 0
        unk = 0
        for cap in captions:
            for tok in _split_caption(cap):
                total += 1
                if tok not in self.word2id:
                    unk += 1
        return 0.0 if total == 0 else unk / total

    # --- persistence ---

    def save(self, dirpath: Path) -> Path:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        out = dirpath / "word_tokenizer.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "name": self.name,
                    "max_length": self.max_length,
                    "word2id": self.word2id,
                },
                f,
            )
        return out

    @classmethod
    def load(cls, dirpath: Path) -> WordTokenizer:
        path = Path(dirpath) / "word_tokenizer.json"
        with open(path) as f:
            data = json.load(f)
        return cls(word2id=data["word2id"], max_length=data["max_length"])


# ─── CLIP BPE wrapper (tokenization only, no embedding weights) ───


class ClipBPETokenizer:
    """Reuses CLIP's BPE vocabulary; encoder embeddings are trained from random init.

    Lazy-loads `transformers.CLIPTokenizerFast` only when first needed so
    importing this module is cheap.
    """

    name = "bpe"

    def __init__(self, max_length: int = 32) -> None:
        from transformers import CLIPTokenizerFast

        self._tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_length
        self.pad_token_id = int(self._tok.pad_token_id or 0)

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        out = self._tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"].long(),
            "attention_mask": out["attention_mask"].long(),
        }

    # --- persistence ---

    def save(self, dirpath: Path) -> Path:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        out = dirpath / "bpe_tokenizer.json"
        with open(out, "w") as f:
            json.dump({"name": self.name, "max_length": self.max_length}, f)
        return out

    @classmethod
    def load(cls, dirpath: Path) -> ClipBPETokenizer:
        path = Path(dirpath) / "bpe_tokenizer.json"
        with open(path) as f:
            data = json.load(f)
        return cls(max_length=data["max_length"])


# ─── Generic loader ───


def load_tokenizer(model_name: str) -> WordTokenizer | ClipBPETokenizer:
    """Resolve `scratch_*_word` or `scratch_*_bpe` -> the saved tokenizer."""
    from src.config import MODELS_DIR

    model_dir = MODELS_DIR / model_name
    if (model_dir / "word_tokenizer.json").exists():
        return WordTokenizer.load(model_dir)
    if (model_dir / "bpe_tokenizer.json").exists():
        return ClipBPETokenizer.load(model_dir)
    raise FileNotFoundError(f"No saved tokenizer in {model_dir}")
