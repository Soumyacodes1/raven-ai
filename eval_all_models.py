"""
Raven AI — Multi-Model Evaluation Script
Tests all 3 epoch models on GoEmotions dataset
Produces a comparison table + confusion matrices for each
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from datasets import load_dataset
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
N_SAMPLES  = 300   # same as evaluate.py for fair comparison
BATCH_SIZE = 32
MAX_LEN    = 128

LABEL2ID = {"happy": 0, "sad": 1, "anxious": 2, "angry": 3, "confused": 4, "neutral": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
RAVEN_EMOTIONS = list(LABEL2ID.keys())

MODEL_PATHS = {
    "Epoch 1": "./emotion_model_epoch1",
    "Epoch 2": "./emotion_model_epoch2",
    "Epoch 3": "./emotion_model_epoch3",
    "Best":    "./emotion_model_best",
}

GO_TO_RAVEN = {
    "joy": "happy", "amusement": "happy", "excitement": "happy",
    "gratitude": "happy", "love": "happy", "optimism": "happy",
    "pride": "happy", "relief": "happy", "admiration": "happy",
    "approval": "happy", "caring": "happy",
    "sadness": "sad", "grief": "sad", "disappointment": "sad",
    "remorse": "sad", "embarrassment": "sad",
    "fear": "anxious", "nervousness": "anxious",
    "anger": "angry", "annoyance": "angry", "disgust": "angry",
    "confusion": "confused", "surprise": "confused",
    "realization": "confused", "curiosity": "confused",
    "neutral": "neutral", "desire": "neutral",
}


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


# ── Load GoEmotions ───────────────────────────────────────────────────────────
def load_goemotions(n_samples=300):
    print("Loading GoEmotions dataset...")
    dataset     = load_dataset("google-research-datasets/go_emotions", "simplified")
    df          = dataset["train"].to_pandas()
    label_names = dataset["train"].features["labels"].feature.names

    rows = []
    for _, row in df.iterrows():
        label_ids = row["labels"]
        if len(label_ids) == 0:
            continue
        go_label    = label_names[int(label_ids[0])]
        raven_label = GO_TO_RAVEN.get(go_label)
        if raven_label:
            rows.append({"text": row["text"], "label": raven_label})

    full_df   = pd.DataFrame(rows)
    per_class = n_samples // len(RAVEN_EMOTIONS)
    balanced  = []
    for emotion in RAVEN_EMOTIONS:
        subset  = full_df[full_df["label"] == emotion]
        sampled = subset.sample(min(per_class, len(subset)), random_state=99)
        balanced.append(sampled)

    result = pd.concat(balanced).sample(frac=1, random_state=99).reset_index(drop=True)
    print(f"Dataset: {len(result)} samples ({per_class} per emotion)")
    return result


# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(model_name, model_path, texts, true_labels):
    if not os.path.exists(model_path):
        print(f"  ⚠ Skipping {model_name} — path not found: {model_path}")
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
    cm  = confusion_matrix(true_labels, pred_labels, labels=RAVEN_EMOTIONS)
    fig, ax = plt.subplots(figsize=(8, 6))
    im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(RAVEN_EMOTIONS)))
    ax.set_yticks(range(len(RAVEN_EMOTIONS)))
    ax.set_xticklabels(RAVEN_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(RAVEN_EMOTIONS)
    ax.set_title(f"Confusion Matrix — {model_name} (Acc: {acc*100:.1f}%)", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(RAVEN_EMOTIONS)):
        for j in range(len(RAVEN_EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fname = f"eval_goemotions_{model_name.lower().replace(' ', '_')}.png"
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


# ── Comparison bar chart ──────────────────────────────────────────────────────
def plot_comparison(results):
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
    ax.set_title("GoEmotions Evaluation — All Epoch Models", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("eval_goemotions_comparison.png", dpi=150)
    plt.close()
    print("\nSaved: eval_goemotions_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🐦‍⬛ Raven AI — GoEmotions Multi-Model Evaluation")
    print("=" * 60)

    df          = load_goemotions(N_SAMPLES)
    texts       = df["text"].tolist()
    true_labels = df["label"].tolist()

    results = []
    for name, path in MODEL_PATHS.items():
        result = evaluate_model(name, path, texts, true_labels)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — GoEmotions Evaluation")
    print("=" * 60)
    summary_df = pd.DataFrame(results)
    summary_df["accuracy"]  = summary_df["accuracy"].map("{:.2%}".format)
    summary_df["precision"] = summary_df["precision"].map("{:.4f}".format)
    summary_df["recall"]    = summary_df["recall"].map("{:.4f}".format)
    summary_df["f1"]        = summary_df["f1"].map("{:.4f}".format)
    print(summary_df.to_string(index=False))

    summary_df.to_csv("eval_goemotions_results.csv", index=False)
    print("\nSaved: eval_goemotions_results.csv")

    # Plot comparison
    results_raw = []
    for name, path in MODEL_PATHS.items():
        r = evaluate_model(name, path, texts, true_labels)
        if r:
            results_raw.append(r)
    plot_comparison(results_raw)

    print("\n✅ GoEmotions evaluation complete!")
    print("Next step: create my_data.csv and run test_my_data.py")
