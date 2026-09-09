import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


TRAIN_FILE = "training_data.csv"
GOLDEN_FILE = "golden_set.csv"

print("=" * 65)
print("HIVER SDE ASSIGNMENT - SEMANTIC INTENT EVALUATION")
print("=" * 65)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

print("\nTraining examples:", len(train_df))
print("Golden evaluation examples:", len(golden_df))


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

train_df = train_df[["text", "intent"]].dropna().copy()
golden_df = golden_df[["text", "intent"]].dropna().copy()

train_df["text"] = train_df["text"].astype(str).str.strip()
train_df["intent"] = train_df["intent"].astype(str).str.strip()

golden_df["text"] = golden_df["text"].astype(str).str.strip()
golden_df["intent"] = golden_df["intent"].astype(str).str.strip()

train_df = train_df[
    (train_df["text"] != "") &
    (train_df["intent"] != "")
]

golden_df = golden_df[
    (golden_df["text"] != "") &
    (golden_df["intent"] != "")
]


# ---------------------------------------------------------
# LOAD SEMANTIC MODEL
# ---------------------------------------------------------

print("\nLoading semantic embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Semantic model loaded.")


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

print("\nCreating training embeddings...")

X_train = model.encode(
    train_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nCreating golden-set embeddings...")

X_test = model.encode(
    golden_df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# TRAIN CLASSIFIER ON EMBEDDINGS
# ---------------------------------------------------------

print("\nTraining Logistic Regression on semantic embeddings...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

classifier.fit(
    X_train,
    train_df["intent"]
)

print("Semantic classifier trained.")


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATING ON UNSEEN GOLDEN SET")
print("=" * 65)

y_true = golden_df["intent"]

y_pred = classifier.predict(X_test)


# ---------------------------------------------------------
# CONFIDENCE
# ---------------------------------------------------------

probabilities = classifier.predict_proba(X_test)

confidence = probabilities.max(axis=1)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\nFINAL RESULTS")
print("-" * 40)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

labels = sorted(
    set(y_true) | set(y_pred)
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nCONFUSION MATRIX")
print("-" * 65)

print(cm_df)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results = golden_df.copy()

results["predicted_intent"] = y_pred
results["confidence"] = confidence

results["correct"] = (
    results["intent"] ==
    results["predicted_intent"]
)

results.to_csv(
    "semantic_evaluation_results.csv",
    index=False
)

print("\nSaved:")
print("semantic_evaluation_results.csv")


# ---------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------

errors = results[
    results["correct"] == False
]

print("\n" + "=" * 65)
print("ERROR ANALYSIS")
print("=" * 65)

print(
    "Incorrect predictions:",
    len(errors)
)


for i, (_, row) in enumerate(
    errors.head(10).iterrows(),
    1
):

    print(f"\nError {i}")

    print("Customer :")
    print(row["text"])

    print("Actual   :", row["intent"])

    print("Predicted:", row["predicted_intent"])

    print(
        "Confidence:",
        f"{row['confidence']:.4f}"
    )


# ---------------------------------------------------------
# CONFIDENCE SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("CONFIDENCE SUMMARY")
print("=" * 65)

print(
    f"Average confidence: {confidence.mean():.4f}"
)

print(
    f"Minimum confidence: {confidence.min():.4f}"
)

print(
    f"Maximum confidence: {confidence.max():.4f}"
)


print("\n" + "=" * 65)
print("SEMANTIC EVALUATION COMPLETE")
print("=" * 65)