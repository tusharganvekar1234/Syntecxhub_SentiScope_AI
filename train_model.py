
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Create data/reviews.csv in the next step.")
        return

    df = pd.read_csv(DATA_PATH)
    required = {"text", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,sentiment")

    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["text"] = df["text"].map(clean_text)
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["text"] != ""]
    df = df[df["sentiment"].isin(["positive", "negative"])]

    if df["sentiment"].nunique() != 2 or df["sentiment"].value_counts().min() < 2:
        raise ValueError("Need at least two examples of each sentiment.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"],
        test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print("\n=== SENTISCOPE AI: MODEL EVALUATION ===")
    print(f"Training samples: {metrics['train_samples']}")
    print(f"Testing samples:  {metrics['test_samples']}")
    print(f"Accuracy:         {metrics['accuracy']:.3f}")
    print(f"Macro F1-score:   {metrics['macro_f1']:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)

    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
