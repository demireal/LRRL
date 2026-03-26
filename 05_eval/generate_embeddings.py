#!/usr/bin/env python3
"""
Generate embeddings from SFT prompts using Qwen3-Embedding-8B.

For each SFT record, this script:
  1. Extracts the user prompt (system + user messages).
  2. Encodes it with Qwen3-Embedding-8B (last-token pooling, L2 norm).
  3. Saves embeddings + metadata as .npz files.

This is used for the embedding-based evaluation approach: the embeddings
serve as features for logistic regression (eval_embeddings.py).

Inputs:
  --sft_dir    : SFT dataset directory (data/sft/{repr}).
  --output_dir : Where to write .npz files.
  --tasks      : Space-separated list (default: all 15).
  --splits     : Which splits to process (default: train val test).

Outputs:
  {output_dir}/{task}/{split}.npz
    Keys: embeddings (N x 4096), patient_ids, labels, prediction_times

Connects to:
  - Upstream  : 02_create_sft, 03_globalrubric, 04_localrubric
  - Downstream: eval_embeddings.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import ALL_TASK_NAMES, EMBEDDING_MODEL, TASKS, SEED


class _Texts(TorchDataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


class EmbeddingEncoder:
    def __init__(self, model_name=EMBEDDING_MODEL, batch_size=12, max_length=8192):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

    @staticmethod
    def _pool(hidden, mask):
        if mask[:, -1].sum() == mask.shape[0]:
            return hidden[:, -1]
        seq_len = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_len]

    @torch.no_grad()
    def encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:\n{t}" for t in texts]
        loader = DataLoader(_Texts(texts), batch_size=self.batch_size, shuffle=False)
        parts = []
        for batch in tqdm(loader, desc="Embedding"):
            tok = self.tokenizer(batch, max_length=self.max_length,
                                 padding=True, truncation=True,
                                 return_tensors="pt").to(self.device)
            out = self.model(**tok)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            emb = self._pool(h, tok["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
            parts.append(emb)
        return np.concatenate(parts, axis=0)


def _extract_prompt(entry: dict) -> str:
    """Get the text that should be embedded (system + user content)."""
    parts = []
    for msg in entry["conversations"][:2]:  # system + user
        parts.append(msg["content"])
    return "\n\n".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--model_name", type=str, default=None,
                   help="Override embedding model (default: from config.tasks)")
    return p.parse_args()


def main():
    args = parse_args()
    # Seed for deterministic inference
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_name = args.model_name if args.model_name else EMBEDDING_MODEL
    encoder = EmbeddingEncoder(model_name=model_name, batch_size=args.batch_size)

    for task in args.tasks:
        query = TASKS.get(task, "")
        for split in args.splits:
            src = Path(args.sft_dir) / split / f"{task}.json"
            if not src.exists():
                continue
            with open(src) as f:
                records = json.load(f)
            if not records:
                continue

            texts = [_extract_prompt(r) for r in records]
            embeddings = encoder.encode(texts, instruction=query)

            patient_ids = np.array([r["patient_id"] for r in records])
            labels = np.array([1 if r["label_value"] else 0 for r in records])
            prediction_times = np.array(
                [r.get("label_time", r.get("prediction_time", "")) for r in records],
                dtype=object,
            )

            out_dir = Path(args.output_dir) / task
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split}.npz"
            np.savez(out_path, embeddings=embeddings,
                     patient_ids=patient_ids, labels=labels,
                     prediction_times=prediction_times)
            logger.info(f"  {task}/{split}: {embeddings.shape} -> {out_path}")

    logger.success("Embedding generation complete.")


if __name__ == "__main__":
    main()
