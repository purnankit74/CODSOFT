# evaluate.py

import pandas as pd

print("Loading prediction file...")
pred_df = pd.read_csv("predictions.csv", dtype={"ID": str})

print("Loading actual (true) data...")
true_df = pd.read_csv(
"test_data_solution.txt",
sep=r"\s*:::\s*",
engine="python",
header=None
)

print("\nSample of raw true data:")
print(true_df.head())

# selecting only ID and GENRE

true_df = true_df[[0, 2]]
true_df.columns = ["ID", "GENRE"]

# cleaning IDs (just in case)

pred_df["ID"] = pred_df["ID"].astype(str).str.strip()
true_df["ID"] = true_df["ID"].astype(str).str.strip()

# merging both

merged_df = pd.merge(pred_df, true_df, on="ID")

print(f"\nTotal matched records: {len(merged_df)}")

# calculating accuracy

correct = (merged_df["PREDICTED_GENRE"] == merged_df["GENRE"]).sum()
total = len(merged_df)

accuracy = correct / total if total > 0 else 0

print(f"\nFinal Accuracy: {accuracy:.4f}")
