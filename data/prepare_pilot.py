import csv
import re
import random
from datasets import load_dataset

DS_NAME = "andythetechnerd03/AI-human-text"
TARGET_HUMAN = 250
TARGET_AI = 250
OUTPUT = "data/raw/pilot_source.csv"
SEED = 42
MIN_WORDS = 50
MAX_WORDS = 400

random.seed(SEED)

HTML_RE = re.compile(r"<[^>]+>|&[a-z]+;", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
MULTI_SPACE = re.compile(r"\s+")

def passes_quality(row):
    text = row["text"] or ""
    if HTML_RE.search(text) or URL_RE.search(text):
        return False
    clean = MULTI_SPACE.sub(" ", text).strip()
    words = clean.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    ascii_ratio = sum(1 for c in clean if ord(c) < 128) / max(len(clean), 1)
    if ascii_ratio < 0.85:
        return False
    return True

def clean_text(text):
    text = HTML_RE.sub("", text)
    text = URL_RE.sub("", text)
    return MULTI_SPACE.sub(" ", text).strip()

ds = load_dataset(DS_NAME, split="train", streaming=True)
ds = ds.shuffle(seed=SEED)

counts = {"0": 0, "1": 0}
collected = []

for row in ds:
    label = str(row["generated"])
    if counts[label] >= TARGET_HUMAN:
        continue
    if not passes_quality(row):
        continue
    collected.append({
        "text": clean_text(row["text"]),
        "label": "human" if label == "0" else "ai",
    })
    counts[label] += 1
    if counts["0"] >= TARGET_HUMAN and counts["1"] >= TARGET_AI:
        break

random.shuffle(collected)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source_id", "text", "label"])
    writer.writeheader()
    for i, row in enumerate(collected):
        writer.writerow({"source_id": i, **row})

print(f"Saved {len(collected)} samples -> {OUTPUT}")
print(f"  human: {counts['0']}   ai: {counts['1']}")
