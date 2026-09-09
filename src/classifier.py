import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the hand-labelled golden set
df = pd.read_csv("golden_set.csv")

X = df["customer_message"].fillna("")
y = df["intent"]

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression classifier
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)

print("\n=== TF-IDF + Logistic Regression ===")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))