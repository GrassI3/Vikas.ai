"""
train.py — Training script for SahaayakTriageNet

Usage:
    python train.py [--epochs 15] [--batch-size 32] [--lr 2e-4] [--device cuda]
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from model.triage_net import SahaayakTriageNet, SEVERITY_CLASSES
from model.data_loader import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
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

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / total

    return total_loss / total, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train SahaayakTriageNet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--include-feedback", action="store_true")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SahaayakTriageNet Training")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # Data
    print("[1/4] Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        include_feedback=args.include_feedback,
    )

    # Model
    print("[2/4] Initializing model...")
    model = SahaayakTriageNet(freeze_encoder=True).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Total params: {total_params:,}")
    print(f"       Trainable:    {trainable:,} ({100*trainable/total_params:.1f}%)")

    # Loss with class weights
    weights_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(len(SEVERITY_CLASSES))],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer — only classification head
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n[3/4] Training...\n")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            model.save_checkpoint(ckpt_path, epoch=epoch, metadata={
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
            })
            print(f"  >>> New best! Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Evaluation on test set
    print(f"\n[4/4] Evaluating on test set...")
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_checkpoint(best_path, device=device)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels, 
        test_preds, 
        target_names=SEVERITY_CLASSES, 
        labels=range(len(SEVERITY_CLASSES)),
        zero_division=0
    ))

    # Save final versioned checkpoint
    version = len([f for f in os.listdir(args.checkpoint_dir) if f.startswith("v")])
    final_path = os.path.join(args.checkpoint_dir, f"v{version + 1}.pt")
    model.save_checkpoint(final_path, epoch=args.epochs, metadata={
        "test_acc": test_acc,
        "version": version + 1,
    })
    print(f"  Saved versioned checkpoint: {final_path}")
    print(f"\n{'='*60}")
    print(f"  Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
