import pandas as pd
import numpy as np

df = pd.read_csv('data/pilot/pilot_dataset.csv')
print(f'Original shape: {df.shape}')
print(f'AI rows: {(df["label"]=="ai").sum()}, Human rows: {(df["label"]=="human").sum()}')
print(f'Level distribution: {df["paraphrase_level"].value_counts().to_dict()}')

# Strategy:
# 1. Select 5,000 AI rows (stratified across L0-L4 levels)
# 2. Select 5,000 Human rows (random)
# 3. Keep remaining AI rows (the "paraphrased text which i want")
#    - Keep all L1-L4 rows
#    - Keep a sample of L0 rows

df = df.copy()

# Step 1: Select 5,000 AI rows (stratified across L0-L4 levels)
ai_rows = df[df['label'] == 'ai']
levels = ['L0', 'L1', 'L2', 'L3', 'L4']
ai_per_level = 5000 // 5  # 1000 per level
ai_sample_list = []
for lv in levels:
    lv_rows = df[(df['label'] == 'ai') & (df['paraphrase_level'] == lv)]
    n = min(len(lv_rows), ai_per_level)
    ai_sample_list.append(df[df['paraphrase_level'] == lv].sample(n=n, random_state=42))
ai_sample = pd.concat(ai_sample_list)

# If total is less than 5000, add more from all AI rows
if len(ai_sample) < 5000:
    additional = df[df['label'] == 'ai'].sample(n=5000 - len(ai_sample), random_state=42)
    ai_sample = pd.concat([ai_sample, additional])

# Step 2: Select 5,000 Human rows (random)
human_rows = df[df['label'] == 'human'].sample(n=5000, random_state=42)

# Step 3: Keep remaining AI rows (the "paraphrased text which i want")
# Keep all L1-L4 rows, and a sample of L0 rows from those not selected
remaining_ai = df[df['label'] == 'ai'].drop(ai_sample['sample_id'].tolist(), errors='ignore')

# Keep all L1-L4 from remaining
remaining_l1l4 = remaining_ai[remaining_ai['paraphrase_level'].isin(['L1', 'L2', 'L3', 'L4'])]

# For L0, keep a sample (max 2000, or whatever is left)
l0_remaining = remaining_ai[remaining_ai['paraphrase_level'] == 'L0']
l0_sample_size = min(2000, len(l0_remaining))
l0_sample = l0_remaining.sample(n=l0_sample_size, random_state=42) if len(l0_remaining) > 2000 else l0_remaining

remaining_ai_thinned = pd.concat([
    remaining_l1l4,
    l0_sample
])

# Shuffle the remaining rows for good measure
remaining_ai_thinned = remaining_ai_thinned.sample(frac=1, random_state=42).reset_index(drop=True)

# === Combine all selected rows ===
result = pd.concat([ai_sample, human_rows, remaining_ai_thinned]).reset_index(drop=True)

print(f'Final shape: {result.shape}')
print(f'AI rows: {(result["label"]=="ai").sum()}')
print(f'Human rows: {(result["label"]=="human").sum()}')
print(f'Level distribution: {result["paraphrase_level"].value_counts().to_dict()}')

# Save
output_path = 'data/pilot/pilot_dataset_thin.csv'
result.to_csv(output_path, index=False)
print(f'Written to {output_path}')
PYEOF