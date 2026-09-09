import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv("golden_set.csv")

X = df["customer_message"].fillna("")
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Majority-class baseline
majority_class = y_train.value_counts().idxmax()

predictions = [majority_class] * len(y_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("=== Majority-Class Baseline ===")
print("Majority class:", majority_class)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))