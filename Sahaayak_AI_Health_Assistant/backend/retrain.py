"""
retrain.py — Dynamic Retraining Pipeline

Reads user-corrected feedback from SQLite, appends to the training set,
runs additional fine-tuning epochs on the classification head, saves a
versioned checkpoint, and marks processed feedback rows.

Usage:
    python retrain.py [--epochs 5] [--batch-size 32] [--lr 1e-4]
"""

import argparse
import os
import sqlite3
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.triage_net import (
    SahaayakTriageNet,
    SEVERITY_CLASSES,
    INDICBERT_MODEL_NAME,
    MAX_SEQ_LENGTH,
)
from model.data_loader import (
    TriageDataset,
    LABEL_TO_IDX,
    get_dataloaders,
)


CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DB_PATH = os.path.join("db", "feedback.sqlite")


def get_pending_feedback() -> list[dict]:
    """Fetch all feedback not yet used in a training epoch."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT id, input_text, user_correction FROM feedback WHERE used_in_epoch IS NULL"
    )
    rows = [
        {"id": row[0], "text": row[1], "label": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return rows


def mark_feedback_used(feedback_ids: list[int], epoch: int):
    """Mark feedback rows as consumed by a training epoch."""
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "UPDATE feedback SET used_in_epoch = ? WHERE id = ?",
        [(epoch, fid) for fid in feedback_ids],
    )
    conn.commit()
    conn.close()


def get_next_version() -> int:
    """Determine the next checkpoint version number."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("v") and f.endswith(".pt")]
    versions = []
    for f in existing:
        try:
            versions.append(int(f[1:].split(".")[0]))
        except ValueError:
            pass
    return max(versions, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Retrain SahaayakTriageNet with feedback")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Sahaayak Dynamic Retraining Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Step 1: Check for pending feedback
    feedback = get_pending_feedback()
    if not feedback:
        print("[RETRAIN] No pending feedback. Nothing to do.")
        return

    print(f"[RETRAIN] Found {len(feedback)} pending feedback corrections")

    # Step 2: Load model from best checkpoint
    model = SahaayakTriageNet(freeze_encoder=True).to(device)
    if os.path.exists(BEST_CHECKPOINT):
        meta = model.load_checkpoint(BEST_CHECKPOINT, device=device)
        print(f"[RETRAIN] Loaded checkpoint: {BEST_CHECKPOINT}")
    else:
        print("[RETRAIN] WARNING: No checkpoint found. Training from scratch.")

    # Step 3: Build feedback dataset
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)
    fb_texts = [f["text"] for f in feedback]
    fb_labels = [LABEL_TO_IDX.get(f["label"], 0) for f in feedback]
    fb_ids = [f["id"] for f in feedback]

    fb_dataset = TriageDataset(fb_texts, fb_labels, tokenizer)
    fb_loader = DataLoader(fb_dataset, batch_size=args.batch_size, shuffle=True)

    # Step 4: Fine-tune
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\n[RETRAIN] Running {args.epochs} fine-tuning epochs on {len(feedback)} samples...\n")
    model.train()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for batch in fb_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {total_loss/total:.4f}  Acc: {correct/total:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    # Step 5: Save versioned checkpoint
    version = get_next_version()
    version_path = os.path.join(CHECKPOINT_DIR, f"v{version}.pt")
    model.save_checkpoint(version_path, epoch=version, metadata={
        "feedback_count": len(feedback),
        "retrained": True,
        "version": version,
    })
    print(f"\n[RETRAIN] Saved versioned checkpoint: {version_path}")

    # Also update best_model.pt
    model.save_checkpoint(BEST_CHECKPOINT, epoch=version, metadata={
        "feedback_count": len(feedback),
        "retrained": True,
        "version": version,
    })
    print(f"[RETRAIN] Updated best checkpoint: {BEST_CHECKPOINT}")

    # Step 6: Mark feedback as used
    mark_feedback_used(fb_ids, version)
    print(f"[RETRAIN] Marked {len(fb_ids)} feedback entries as used in epoch v{version}")

    print(f"\n{'='*60}")
    print(f"  Retraining complete! Version: v{version}")
    print(f"  Call POST /api/reload to hot-swap the model.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
