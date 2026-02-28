import os
import json
import pandas as pd

BASE_DIR = os.path.abspath("../code/fakenewsnet_dataset")
rows = []

def collect(source, label):
    path = os.path.join(BASE_DIR, source, label)

    if not os.path.exists(path):
        print(f"Skipping missing path: {path}")
        return

    for item in os.listdir(path):
        news_path = os.path.join(path, item, "news content.json")
        if not os.path.exists(news_path):
            continue

        try:
            with open(news_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        text = data.get("text", "").strip()
        if len(text) < 100:
            continue

        rows.append({
            "text": text,
            "label": 1 if label == "fake" else 0
        })

for src in ["politifact", "gossipcop"]:
    for lbl in ["fake", "real"]:
        collect(src, lbl)

df = pd.DataFrame(rows)
df.to_csv("data/fakenews.csv", index=False)

print("Dataset size:", len(df))
print(df["label"].value_counts())
