# test.py

import pandas as pd
import pickle

print("Loading model and vectorizer...")

with open("model.pkl", "rb") as f:
 model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
 vectorizer = pickle.load(f)

print("Reading test data...")

test_data = pd.read_csv(
"test_data.txt",
sep=" ::: ",
engine="python",
names=["ID", "TITLE", "DESCRIPTION"]
)

# clean text

test_data["DESCRIPTION"] = test_data["DESCRIPTION"].astype(str).str.lower()

print("Transforming text...")
X_test = vectorizer.transform(test_data["DESCRIPTION"])

print("Making predictions...")
preds = model.predict(X_test)

# preparing output file

output = pd.DataFrame({
"ID": test_data["ID"].astype(str),
"PREDICTED_GENRE": preds
})

output.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv\n")
print(output.head())
