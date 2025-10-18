import pandas as pd

# Load both tables
basic = pd.read_excel("MMASD basic - clean.xlsx")
feats = pd.read_excel("MMASD_features.xlsx")

print("Basic columns:", basic.columns.tolist())
print("Features columns:", feats.columns.tolist())

# Merge on sample_id (basic) and clip_id (features)
merged = pd.merge(
    basic,
    feats,
    left_on="sample_id",
    right_on="clip_id",
    how="outer",
    suffixes=("_basic", "_feat")
)

# Save the combined table
merged.to_excel("MMASD_merged.xlsx", index=False)
print("Merged table saved. Shape:", merged.shape)

# Report
only_basic = merged[merged["clip_id"].isna()]
only_feats = merged[merged["sample_id"].isna()]
matched = merged.dropna(subset=["sample_id", "clip_id"])

print("\n--- Merge Report ---")
print("Total rows in merged table:", len(merged))
print("Matched rows (in both):", len(matched))
print("Only in basic table:", len(only_basic))
print("Only in features table:", len(only_feats))
