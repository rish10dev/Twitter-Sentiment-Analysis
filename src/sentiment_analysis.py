import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

# Rename Columns
df.columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "tweet"
]

print("Dataset Shape:")
print(df.shape)

# Use smaller sample for faster training
df = df.sample(50000, random_state=42)

# Convert sentiment
df["sentiment"] = df["sentiment"].replace(4, 1)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Apply Cleaning
df["clean_tweet"] = df["tweet"].apply(clean_text)

# Sentiment Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("images/sentiment_distribution.png")
plt.close()

# Features and Labels
X = df["clean_tweet"]
y = df["sentiment"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "models/sentiment_model.pkl")

# Save Vectorizer
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")
print("Graph saved in images folder.")