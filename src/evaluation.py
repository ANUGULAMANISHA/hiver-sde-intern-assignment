import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


TRAIN_FILE = "training_data.csv"
GOLDEN_FILE = "golden_set.csv"


print("=" * 65)
print("HIVER SDE ASSIGNMENT - UNSEEN GOLDEN SET EVALUATION")
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


# ---------------------------------------------------------
# REMOVE EMPTY ROWS
# ---------------------------------------------------------

train_df = train_df[
    (train_df["text"] != "") &
    (train_df["intent"] != "")
]

golden_df = golden_df[
    (golden_df["text"] != "") &
    (golden_df["intent"] != "")
]


# ---------------------------------------------------------
# VERIFY INTENTS
# ---------------------------------------------------------

train_intents = set(train_df["intent"].unique())
golden_intents = set(golden_df["intent"].unique())

print("\nTraining intents:")
for intent in sorted(train_intents):
    print(" -", intent)

print("\nGolden intents:")
for intent in sorted(golden_intents):
    print(" -", intent)


unknown = golden_intents - train_intents

if unknown:
    print("\nWARNING: Golden intents missing from training:")
    for intent in unknown:
        print(" -", intent)


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING TF-IDF + LOGISTIC REGRESSION")
print("=" * 65)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ---------------------------------------------------------
# TRAIN ONLY ON HISTORICAL DATA
# ---------------------------------------------------------

model.fit(
    train_df["text"],
    train_df["intent"]
)

print("Model trained successfully.")


# ---------------------------------------------------------
# EVALUATE ON UNSEEN GOLDEN SET
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATING ON 200 UNSEEN GOLDEN EXAMPLES")
print("=" * 65)

y_true = golden_df["intent"]

y_pred = model.predict(
    golden_df["text"]
)


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

results["correct"] = (
    results["intent"] ==
    results["predicted_intent"]
)

results.to_csv(
    "golden_evaluation_results.csv",
    index=False
)

print("\nSaved:")
print("golden_evaluation_results.csv")


# ---------------------------------------------------------
# SHOW ERRORS
# ---------------------------------------------------------

errors = results[
    results["correct"] == False
]

print("\n" + "=" * 65)
print("ERROR ANALYSIS")
print("=" * 65)

print("Incorrect predictions:", len(errors))

for i, (_, row) in enumerate(errors.head(10).iterrows(), 1):

    print(f"\nError {i}")
    print("Customer :", row["text"])
    print("Actual   :", row["intent"])
    print("Predicted:", row["predicted_intent"])


print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)