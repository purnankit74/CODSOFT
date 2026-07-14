# train.py

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

print("Loading training data...")
data = pd.read_csv(
"train_data.txt",
sep=" ::: ",
engine="python",
names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# basic cleaning

data["DESCRIPTION"] = data["DESCRIPTION"].astype(str).str.lower()

# features and labels

X = data["DESCRIPTION"]
y = data["GENRE"]

print("Converting text to numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = vectorizer.fit_transform(X)

# split data

X_train, X_test, y_train, y_test = train_test_split(
X_vec, y, test_size=0.2, random_state=42
)

# trying multiple models (just to see which works better)

models = [
("Naive Bayes", MultinomialNB()),
("Logistic Regression", LogisticRegression(max_iter=200)),
("SVM", LinearSVC())
]

best_model = None
best_score = 0

print("\nTraining models...\n")

for name, model in models:
 print(f"Training {name}...")

model.fit(X_train, y_train)
preds = model.predict(X_test)

score = accuracy_score(y_test, preds)
print(f"{name} Accuracy: {score:.4f}\n")

if score > best_score:
    best_score = score
    best_model = model
print("Best model selected!")

# saving model and vectorizer

with open("model.pkl", "wb") as f:
 pickle.dump(best_model, f)

with open("tfidf.pkl", "wb") as f:
 pickle.dump(vectorizer, f)

print("Files saved successfully!")
