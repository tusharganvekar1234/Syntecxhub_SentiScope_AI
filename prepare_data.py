
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

files = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt",
]

datasets = []

for filename in files:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        quoting=3,
        encoding="utf-8"
    )

    df["source"] = filename.replace("_labelled.txt", "")
    datasets.append(df)

data = pd.concat(datasets, ignore_index=True)

data["text"] = data["text"].astype(str).str.strip()
data["label"] = pd.to_numeric(data["label"], errors="coerce")

data = data.dropna(subset=["text", "label"])
data = data[data["label"].isin([0, 1])]
data = data[data["text"] != ""]

data["sentiment"] = data["label"].map({
    0: "negative",
    1: "positive"
})

output = data[["text", "sentiment", "source"]]
output.to_csv(DATA_DIR / "reviews.csv", index=False)

print("Dataset prepared successfully!")
print(f"Total reviews: {len(output)}")
print("\nSentiment distribution:")
print(output["sentiment"].value_counts())
print("\nSource distribution:")
print(output["source"].value_counts())
print(f"\nSaved to: {DATA_DIR / 'reviews.csv'}")
