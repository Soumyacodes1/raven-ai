"""
Raven AI — Personal Dataset Test Script
Tests all 3 epoch models + best model on your own labelled data
Combines my_dataset_1.csv through my_dataset_4.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
MAX_LEN    = 128

LABEL2ID = {"happy": 0, "sad": 1, "anxious": 2, "angry": 3, "confused": 4, "neutral": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
RAVEN_EMOTIONS = list(LABEL2ID.keys())

MODEL_PATHS = {
    "Original":           "./emotion_model",
    "Epoch 1":            "./emotion_model_epoch1",
    "Epoch 2":            "./emotion_model_epoch2",
    "Epoch 3":            "./emotion_model_epoch3",
    "Best":               "./emotion_model_best",
    "Continued Epoch 1":  "./emotion_model_continued_epoch1",
    "Continued Epoch 2":  "./emotion_model_continued_epoch2",
    "Continued Epoch 3":  "./emotion_model_continued_epoch3",
    "Continued Epoch 4":  "./emotion_model_continued_epoch4",
    "Continued Epoch 5":  "./emotion_model_continued_epoch5",
    "Continued Best":     "./emotion_model_continued_best",
}

MY_DATA_FILES = [
    "my_dataset_1.csv",
    "my_dataset_2.csv",
    "my_dataset_3.csv",
    "my_dataset_4.csv",
]


# ── Dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Load personal data ────────────────────────────────────────────────────────
def load_my_data():
    print("Loading your personal datasets...")
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


# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(model_name, model_path, texts, true_labels):
    if not os.path.exists(model_path):
        print(f"  Skipping {model_name} — path not found: {model_path}")
        return None

    print(f"\nEvaluating: {model_name} ({model_path})")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    label_ids = [LABEL2ID[l] for l in true_labels]
    dataset   = EmotionDataset(texts, label_ids, tokenizer)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    pred_labels = [ID2LABEL[p] for p in all_preds]

    acc  = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    rec  = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1   = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, pred_labels,
                                target_names=RAVEN_EMOTIONS, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=RAVEN_EMOTIONS)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(RAVEN_EMOTIONS)))
    ax.set_yticks(range(len(RAVEN_EMOTIONS)))
    ax.set_xticklabels(RAVEN_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(RAVEN_EMOTIONS)
    ax.set_title(f"My Data — {model_name} (Acc: {acc*100:.1f}%)", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(RAVEN_EMOTIONS)):
        for j in range(len(RAVEN_EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fname = f"mydata_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

    return {
        "model":     model_name,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }


# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(results, goemotions_results=None):
    models  = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#6366F1", "#10B981", "#F59E0B", "#F87171"]

    x     = range(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar([xi + i * width for xi in x], vals, width,
                      label=label, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("My Data Evaluation — All Epoch Models", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("mydata_comparison.png", dpi=150)
    plt.close()
    print("Saved: mydata_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🐦‍⬛ Raven AI — Personal Dataset Evaluation")
    print("=" * 60)

    df = load_my_data()
    if df is None:
        exit()

    texts       = df["text"].tolist()
    true_labels = df["label"].tolist()

    print(f"\n{'='*60}")
    print("Testing all models on your personal data...")
    print("=" * 60)

    results = []
    for name, path in MODEL_PATHS.items():
        result = evaluate_model(name, path, texts, true_labels)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — Your Personal Data")
    print("=" * 60)
    summary_df = pd.DataFrame(results)
    summary_df["accuracy"]  = summary_df["accuracy"].map("{:.2%}".format)
    summary_df["precision"] = summary_df["precision"].map("{:.4f}".format)
    summary_df["recall"]    = summary_df["recall"].map("{:.4f}".format)
    summary_df["f1"]        = summary_df["f1"].map("{:.4f}".format)
    print(summary_df.to_string(index=False))

    summary_df.to_csv("mydata_results.csv", index=False)
    print("\nSaved: mydata_results.csv")

    plot_comparison(results)

    # Final recommendation
    raw_results = [r for r in results if isinstance(r.get("accuracy"), float)]
    if raw_results:
        best = max(raw_results, key=lambda x: x["accuracy"])
        print(f"\n🏆 Best model on your data: {best['model']} ({best['accuracy']*100:.2f}%)")
        print("\nNext step: update emotion_engine.py to use the best model path")
