"""
Raven AI — Continued Training Script
Fine-tunes emotion_model_best/ on personal datasets (my_dataset_1-4.csv)
Goal: improve personal data accuracy from 64.62% toward 75%+
Output: ./emotion_model_continued_epoch{1..5}/ ./emotion_model_continued_best/
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────────────────
N_EPOCHS    = 5
BATCH_SIZE  = 16
MAX_LEN     = 128
LR          = 5e-6       # quarter of original 2e-5 — prevents catastrophic forgetting
PATIENCE    = 2          # early stopping patience (epochs without val accuracy improvement)

BASE_MODEL  = "./emotion_model_best"
SAVE_BASE   = "./emotion_model_continued"
BEST_PATH   = "./emotion_model_continued_best"

LABEL2ID = {"happy": 0, "sad": 1, "anxious": 2, "angry": 3, "confused": 4, "neutral": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
RAVEN_EMOTIONS = list(LABEL2ID.keys())

MY_DATA_FILES = [
    "my_dataset_1.csv",
    "my_dataset_2.csv",
    "my_dataset_3.csv",
    "my_dataset_4.csv",
]


# ── Dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Load personal data ───────────────────────────────────────────────────────
def load_personal_data():
    print("Loading personal datasets...")
    dfs = []
    for fname in MY_DATA_FILES:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            dfs.append(df)
            print(f"  Loaded {fname}: {len(df)} rows")
        else:
            print(f"  Skipping {fname} — not found")

    if not dfs:
        print("No dataset files found!")
        return None

    combined = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

    # Validate labels
    valid = combined["label"].isin(RAVEN_EMOTIONS)
    invalid = combined[~valid]
    if len(invalid) > 0:
        print(f"\nWarning: {len(invalid)} rows with invalid labels removed:")
        print(invalid["label"].value_counts())
    combined = combined[valid].reset_index(drop=True)

    print(f"\nFinal dataset: {len(combined)} unique rows")
    print(combined["label"].value_counts().to_string())
    return combined


# ── Evaluate one epoch ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            total_loss    += outputs.loss.item()
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    avg_loss    = total_loss / len(loader)
    pred_labels = [ID2LABEL[p] for p in all_preds]
    true_labels = [ID2LABEL[l] for l in all_labels]
    acc  = accuracy_score(true_labels, pred_labels)
    f1   = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    prec = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    rec  = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    return acc, f1, prec, rec, avg_loss, pred_labels, true_labels


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_cm(true_labels, pred_labels, title, filename):
    cm = confusion_matrix(true_labels, pred_labels, labels=RAVEN_EMOTIONS)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(RAVEN_EMOTIONS)))
    ax.set_yticks(range(len(RAVEN_EMOTIONS)))
    ax.set_xticklabels(RAVEN_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(RAVEN_EMOTIONS)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(RAVEN_EMOTIONS)):
        for j in range(len(RAVEN_EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ── Main training ─────────────────────────────────────────────────────────────
def train():
    print("\n🐦‍⬛ Raven AI — Continued Training on Personal Data")
    print("=" * 60)

    df = load_personal_data()
    if df is None:
        return

    texts  = df["text"].tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)} | Validation: {len(X_val)}")

    # Load the fine-tuned model as starting checkpoint
    print(f"\nLoading base model from {BASE_MODEL}/...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    train_dataset = EmotionDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset   = EmotionDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * N_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    print(f"\nStarting continued training — up to {N_EPOCHS} epochs (patience={PATIENCE})")
    print(f"Learning rate: {LR} (reduced from 2e-5 to prevent forgetting)")
    print("=" * 60)

    epoch_results = []
    train_losses  = []
    val_accs      = []
    best_acc      = 0
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss  = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{N_EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate
        acc, f1, prec, rec, val_loss, pred_labels, true_labels = evaluate(model, val_loader, device)
        val_accs.append(acc)
        epoch_time = time.time() - epoch_start

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} COMPLETE — {epoch_time:.1f}s")
        print(f"  Train Loss:    {avg_train_loss:.4f}")
        print(f"  Val Loss:      {val_loss:.4f}")
        print(f"  Val Accuracy:  {acc*100:.2f}%")
        print(f"  Val F1:        {f1:.4f}")
        print(f"  Val Precision: {prec:.4f}")
        print(f"  Val Recall:    {rec:.4f}")

        # Save this epoch's model
        epoch_path = f"{SAVE_BASE}_epoch{epoch+1}"
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
        print(f"  Saved: {epoch_path}/")

        # Save confusion matrix
        plot_cm(
            true_labels, pred_labels,
            f"Continued Training — Epoch {epoch+1} (Acc: {acc*100:.1f}%)",
            f"continued_confusion_matrix_epoch{epoch+1}.png"
        )

        # Classification report
        report = classification_report(
            true_labels, pred_labels,
            target_names=RAVEN_EMOTIONS, zero_division=0
        )
        print(f"\nClassification Report — Epoch {epoch+1}:")
        print(report)

        epoch_results.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "val_precision": prec,
            "val_recall": rec,
        })

        # Track best model
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            model.save_pretrained(BEST_PATH)
            tokenizer.save_pretrained(BEST_PATH)
            print(f"  🏆 New best model saved to {BEST_PATH}/ ({acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        print(f"{'='*60}\n")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {PATIENCE} epochs)")
            break

    # Save results table
    actual_epochs = len(epoch_results)
    results_df = pd.DataFrame(epoch_results)
    results_df.to_csv("continued_training_results.csv", index=False)
    print("Saved: continued_training_results.csv")
    print(results_df.to_string(index=False))

    # Plot training curve
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    epochs_range = range(1, actual_epochs + 1)
    ax1.plot(epochs_range, train_losses, marker="o",
             label="Train Loss", color="#6366F1", linewidth=2)
    ax2.plot(epochs_range, val_accs, marker="s",
             label="Val Accuracy", color="#10B981", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#6366F1")
    ax2.set_ylabel("Val Accuracy", color="#10B981")
    ax1.set_title("Raven — Continued Training on Personal Data", fontsize=13)
    ax1.set_xticks(list(epochs_range))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    plt.savefig("continued_training_curve.png", dpi=150)
    plt.close()
    print("Saved: continued_training_curve.png")

    print(f"\n🐦‍⬛ Continued training complete!")
    print(f"   Best accuracy: {best_acc*100:.2f}%")
    print(f"   Models saved:")
    for i in range(actual_epochs):
        r = epoch_results[i]
        print(f"   → emotion_model_continued_epoch{i+1}/ — Acc: {r['val_accuracy']*100:.2f}% | F1: {r['val_f1']:.4f}")
    print(f"   → emotion_model_continued_best/   — Best overall model")
    print(f"\nNext step: run 'python test_my_data.py' to verify improvement on personal data")
    print(f"Then update config.py: HF_EMOTION_MODEL = './emotion_model_continued_best'")


if __name__ == "__main__":
    train()
