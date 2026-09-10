import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.hybrid_classifier import hybrid_predict


# ============================================================
# Hiver SDE Assignment
# Hybrid Classifier Evaluation
#
# IMPORTANT:
# - golden_set.csv is ONLY used for evaluation
# - training happens inside hybrid_classifier.py
# - no Gemini API is used
# ============================================================

GOLDEN_FILE = "golden_set.csv"
RESULT_FILE = "hybrid_evaluation_results.csv"


# ------------------------------------------------------------
# Load golden set
# ------------------------------------------------------------

def load_golden_set():

    df = pd.read_csv(
        GOLDEN_FILE
    )

    required_columns = [
        "customer_message",
        "intent"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "customer_message",
            "intent"
        ]
    ).copy()

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    return df


# ------------------------------------------------------------
# Run hybrid evaluation
# ------------------------------------------------------------

def run_evaluation():

    print("=" * 70)
    print("HIVER SDE ASSIGNMENT - HYBRID CLASSIFIER EVALUATION")
    print("=" * 70)

    golden = load_golden_set()

    print(
        f"\nGolden examples: {len(golden)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The golden set is used only for evaluation."
    )

    print(
        "The classifier is trained using training_data.csv."
    )

    print(
        "No Gemini API calls are used."
    )

    print(
        "\nRunning predictions..."
    )

    results = []

    for position, row in golden.iterrows():

        customer_message = (
            row["customer_message"]
        )

        expected_intent = (
            row["intent"]
        )

        try:

            prediction = hybrid_predict(
                customer_message
            )

            predicted_intent = (
                prediction["intent"]
            )

            confidence = (
                prediction["confidence"]
            )

            tfidf_intent = (
                prediction["tfidf_intent"]
            )

            tfidf_confidence = (
                prediction["tfidf_confidence"]
            )

            semantic_intent = (
                prediction["semantic_intent"]
            )

            semantic_confidence = (
                prediction["semantic_confidence"]
            )

            results.append(
                {
                    "customer_message":
                        customer_message,

                    "gold_intent":
                        expected_intent,

                    "predicted_intent":
                        predicted_intent,

                    "intent_correct":
                        int(
                            expected_intent
                            == predicted_intent
                        ),

                    "hybrid_confidence":
                        confidence,

                    "tfidf_intent":
                        tfidf_intent,

                    "tfidf_confidence":
                        tfidf_confidence,

                    "semantic_intent":
                        semantic_intent,

                    "semantic_confidence":
                        semantic_confidence
                }
            )

        except Exception as error:

            print(
                f"\nPrediction error at example "
                f"{position + 1}: {error}"
            )

            results.append(
                {
                    "customer_message":
                        customer_message,

                    "gold_intent":
                        expected_intent,

                    "predicted_intent":
                        "",

                    "intent_correct":
                        "",

                    "hybrid_confidence":
                        "",

                    "tfidf_intent":
                        "",

                    "tfidf_confidence":
                        "",

                    "semantic_intent":
                        "",

                    "semantic_confidence":
                        ""
                }
            )

        if (
            (position + 1) % 10 == 0
            or position + 1 == len(golden)
        ):

            print(
                f"Processed "
                f"{position + 1}/{len(golden)}"
            )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Keep only successful predictions
    # --------------------------------------------------------

    valid = results_df[
        results_df["predicted_intent"]
        .astype(str)
        .str.strip()
        != ""
    ].copy()

    if len(valid) == 0:

        raise RuntimeError(
            "No successful predictions were generated."
        )

    y_true = (
        valid["gold_intent"]
        .astype(str)
    )

    y_pred = (
        valid["predicted_intent"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print headline results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID CLASSIFIER RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluation examples: "
        f"{len(valid)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
        f" ({accuracy:.2%})"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    confidence_values = pd.to_numeric(
        valid["hybrid_confidence"],
        errors="coerce"
    )

    print(
        "\nHybrid confidence:"
    )

    print(
        f"Average: "
        f"{confidence_values.mean():.4f}"
    )

    print(
        f"Minimum: "
        f"{confidence_values.min():.4f}"
    )

    print(
        f"Maximum: "
        f"{confidence_values.max():.4f}"
    )

    # --------------------------------------------------------
    # Per-intent classification report
    # --------------------------------------------------------

    print(
        "\nPer-intent classification report:"
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        golden["intent"].unique()
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_df
    )

    # --------------------------------------------------------
    # Intent-level accuracy
    # --------------------------------------------------------

    print(
        "\nIntent-level accuracy:"
    )

    for intent in labels:

        intent_rows = valid[
            valid["gold_intent"]
            == intent
        ]

        if len(intent_rows) == 0:
            continue

        intent_accuracy = (
            intent_rows["intent_correct"]
            .astype(int)
            .mean()
        )

        print(
            f"{intent}: "
            f"{intent_accuracy:.2%} "
            f"({len(intent_rows)} examples)"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        f"\nDetailed results saved to: "
        f"{RESULT_FILE}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_evaluation()
