"""
Raven AI — Evaluation Script
Experiments:
  1. Zero-shot emotion classification (no examples given)
  2. Few-shot emotion classification (3 examples given)
Dataset: GoEmotions (Google) — 1000 balanced samples
"""

import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from groq import Groq
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from config import GROQ_API_KEY, GROQ_MODEL_EVAL

client = Groq(api_key=GROQ_API_KEY)

# ── Emotion mapping: GoEmotions 28 → Raven 6 ─────────────────────────────────
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

RAVEN_EMOTIONS = ["happy", "sad", "anxious", "angry", "confused", "neutral"]

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    ("I just got promoted at work, this is the best day ever!", "happy"),
    ("I can't stop crying, I feel completely alone in this world.", "sad"),
    ("My heart is racing, I don't know if I can handle this presentation.", "anxious"),
    ("This is absolutely ridiculous, I can't believe they did that!", "angry"),
    ("Wait, so does that mean the meeting is today or tomorrow? I'm so lost.", "confused"),
    ("I went to the store and bought some groceries.", "neutral"),
]

# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_per_minute=28):
        self.max_per_minute = max_per_minute
        self.calls = []

    def wait(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 1
            print(f"  Rate limit reached — waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        self.calls.append(time.time())

rate_limiter = RateLimiter(max_per_minute=28)


# ── Load & prepare dataset ────────────────────────────────────────────────────
def load_goemotions(n_samples=1000):
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    df = dataset["train"].to_pandas()

    label_names = dataset["train"].features["labels"].feature.names

    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        label_ids = row["labels"]
        if len(label_ids) == 0:
            continue
        go_label = label_names[int(label_ids[0])]
        raven_label = GO_TO_RAVEN.get(go_label)
        if raven_label:
            rows.append({"text": text, "true_label": raven_label})

    full_df = pd.DataFrame(rows)

    # Balance classes — equal samples per emotion
    per_class = n_samples // len(RAVEN_EMOTIONS)
    balanced = []
    for emotion in RAVEN_EMOTIONS:
        subset = full_df[full_df["true_label"] == emotion]
        sampled = subset.sample(min(per_class, len(subset)), random_state=42)
        balanced.append(sampled)

    result = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset ready: {len(result)} samples across {len(RAVEN_EMOTIONS)} emotions")
    print(result["true_label"].value_counts().to_string())
    return result


# ── Zero-shot classifier ──────────────────────────────────────────────────────
def zero_shot_classify(text):
    rate_limiter.wait()
    prompt = f"""You are an emotion classifier. Classify the emotional tone of the text below into exactly one of these emotions: happy, sad, anxious, angry, confused, neutral.

Rules:
- Reply with ONLY the emotion word, nothing else
- No punctuation, no explanation
- If unsure, choose neutral

Text: "{text}"

Emotion:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_EVAL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        result = ''.join(c for c in result if c.isalpha())
        return result if result in RAVEN_EMOTIONS else "neutral"
    except Exception as e:
        print(f"  Error: {e} — defaulting to neutral")
        time.sleep(5)
        return "neutral"


# ── Few-shot classifier ───────────────────────────────────────────────────────
def few_shot_classify(text):
    rate_limiter.wait()
    examples_text = "\n".join([
        f'Text: "{t}"\nEmotion: {e}' for t, e in FEW_SHOT_EXAMPLES
    ])
    prompt = f"""You are an emotion classifier. Here are some examples:

{examples_text}

Now classify this text into exactly one of: happy, sad, anxious, angry, confused, neutral.
Reply with ONLY the emotion word, nothing else.

Text: "{text}"

Emotion:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_EVAL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        result = ''.join(c for c in result if c.isalpha())
        return result if result in RAVEN_EMOTIONS else "neutral"
    except Exception as e:
        print(f"  Error: {e} — defaulting to neutral")
        time.sleep(5)
        return "neutral"


# ── Run experiment ────────────────────────────────────────────────────────────
def run_experiment(df, experiment_name, classify_fn):
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}")

    predictions = []
    total = len(df)

    for i, row in df.iterrows():
        pred = classify_fn(row["text"])
        predictions.append(pred)

        if (i + 1) % 50 == 0:
            correct = sum(p == t for p, t in zip(predictions, df["true_label"][:len(predictions)]))
            running_acc = correct / len(predictions) * 100
            print(f"  Progress: {len(predictions)}/{total} | Running accuracy: {running_acc:.1f}%")

    return predictions


# ── Compute & display metrics ─────────────────────────────────────────────────
def compute_metrics(true_labels, predictions, experiment_name):
    acc  = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    rec  = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1   = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    print(f"\n── {experiment_name} Results ──")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(true_labels, predictions,
                                 target_names=RAVEN_EMOTIONS, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(true_labels, predictions, experiment_name, filename):
    cm = confusion_matrix(true_labels, predictions, labels=RAVEN_EMOTIONS)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(RAVEN_EMOTIONS)))
    ax.set_yticks(range(len(RAVEN_EMOTIONS)))
    ax.set_xticklabels(RAVEN_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(RAVEN_EMOTIONS)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    for i in range(len(RAVEN_EMOTIONS)):
        for j in range(len(RAVEN_EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ── Plot comparison bar chart ─────────────────────────────────────────────────
def plot_comparison(zs_metrics, fs_metrics):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    zs_vals = [zs_metrics[m] for m in metrics]
    fs_vals = [fs_metrics[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], zs_vals, width,
                   label="Zero-Shot", color="#6366F1", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], fs_vals, width,
                   label="Few-Shot", color="#10B981", alpha=0.85)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Zero-Shot vs Few-Shot — Raven Emotion Classifier", fontsize=13)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=150)
    plt.close()
    print("  Saved: comparison_chart.png")


# ── Save results to CSV ───────────────────────────────────────────────────────
def save_results(df, zs_preds, fs_preds):
    df["zero_shot_pred"] = zs_preds
    df["few_shot_pred"]  = fs_preds
    df["zs_correct"]     = df["zero_shot_pred"] == df["true_label"]
    df["fs_correct"]     = df["few_shot_pred"]  == df["true_label"]
    df.to_csv("evaluation_results.csv", index=False)
    print("  Saved: evaluation_results.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🐦‍⬛ Raven AI — Emotion Classification Evaluation")
    print("="*60)

    # Load dataset
    df = load_goemotions(n_samples=300)

    # Experiment 1 — Zero-shot
    zs_preds = run_experiment(df, "Experiment 1: Zero-Shot", zero_shot_classify)
    zs_metrics = compute_metrics(df["true_label"].tolist(), zs_preds, "Zero-Shot")
    plot_confusion_matrix(df["true_label"].tolist(), zs_preds,
                          "Zero-Shot", "confusion_matrix_zero_shot.png")

    # Experiment 2 — Few-shot
    fs_preds = run_experiment(df, "Experiment 2: Few-Shot", few_shot_classify)
    fs_metrics = compute_metrics(df["true_label"].tolist(), fs_preds, "Few-Shot")
    plot_confusion_matrix(df["true_label"].tolist(), fs_preds,
                          "Few-Shot", "confusion_matrix_few_shot.png")

    # Comparison
    print("\n── Comparison Summary ──")
    print(f"  Zero-Shot Accuracy: {zs_metrics['accuracy']*100:.2f}%")
    print(f"  Few-Shot  Accuracy: {fs_metrics['accuracy']*100:.2f}%")
    improvement = (fs_metrics['accuracy'] - zs_metrics['accuracy']) * 100
    print(f"  Improvement:        {improvement:+.2f}%")
    plot_comparison(zs_metrics, fs_metrics)

    # Save CSV
    save_results(df, zs_preds, fs_preds)

    print("\n✅ Evaluation complete! Files saved:")
    print("   confusion_matrix_zero_shot.png")
    print("   confusion_matrix_few_shot.png")
    print("   comparison_chart.png")
    print("   evaluation_results.csv")
