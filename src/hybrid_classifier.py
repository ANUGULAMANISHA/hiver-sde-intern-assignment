import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Hiver SDE Assignment - Hybrid Intent Classifier
# ============================================================
#
# Pipeline:
# 1. TF-IDF + Logistic Regression
# 2. Semantic similarity using SentenceTransformer
# 3. Weighted hybrid scoring
# 4. Lightweight high-precision intent guardrails
#
# IMPORTANT:
# - training_data.csv is used for training
# - golden_set.csv is NOT used here
# - golden_set.csv is reserved for evaluation
# ============================================================


# ------------------------------------------------------------
# 1. Load historical training data
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "training_data.csv"

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"training_data.csv not found at: {TRAIN_FILE}\n"
        "Place training_data.csv in the project root directory."
    )


df = pd.read_csv(TRAIN_FILE)


# The prepared training file uses "text".
# Rename it for consistency.
if (
    "customer_message" not in df.columns
    and "text" in df.columns
):
    df = df.rename(
        columns={
            "text": "customer_message"
        }
    )


# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required_columns = [
    "customer_message",
    "intent"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# ------------------------------------------------------------
# Clean training data
# ------------------------------------------------------------

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


df = df[
    (df["customer_message"] != "")
    &
    (df["intent"] != "")
].copy()


# ------------------------------------------------------------
# Display dataset information
# ------------------------------------------------------------

print("=" * 65)
print("HIVER SDE ASSIGNMENT - HYBRID INTENT CLASSIFIER")
print("=" * 65)

print(
    f"\nTraining examples: {len(df)}"
)

print(
    f"Intents: {df['intent'].nunique()}"
)


labels = sorted(
    df["intent"].unique()
)


print("\nIntents:")

for label in labels:
    print(
        f"- {label}"
    )


print("\nIntent distribution:")

print(
    df["intent"].value_counts()
)


# ============================================================
# 2. TF-IDF + Logistic Regression
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "TRAINING TF-IDF + LOGISTIC REGRESSION"
)

print(
    "=" * 65
)


tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)


X_tfidf = tfidf.fit_transform(
    df["customer_message"]
)


tfidf_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


tfidf_model.fit(
    X_tfidf,
    df["intent"]
)


print(
    "TF-IDF model trained successfully."
)


# ============================================================
# 3. Semantic Embeddings
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "LOADING SEMANTIC MODEL"
)

print(
    "=" * 65
)


semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Creating semantic embeddings..."
)


embeddings = semantic_model.encode(
    df["customer_message"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


print(
    "Semantic embeddings created."
)


print(
    f"Embedding count: {len(embeddings)}"
)


# ============================================================
# 4. Text normalization
# ============================================================

def normalize_message(message):
    """
    Normalize a customer message for lightweight
    rule-based guardrails.
    """

    message = str(message).lower().strip()

    # Replace punctuation with spaces
    message = re.sub(
        r"[^a-z0-9\s]",
        " ",
        message
    )

    # Collapse repeated spaces
    message = re.sub(
        r"\s+",
        " ",
        message
    )

    return message


# ============================================================
# 5. Intent guardrails
# ============================================================

def apply_intent_guardrails(
    message,
    hybrid_intent,
    hybrid_confidence
):
    """
    Apply small high-precision rules after the
    statistical hybrid classifier.

    These rules are intentionally conservative.

    Priority:

    1. Late / Missing Delivery
    2. Order Tracking / Status

    This prevents common tracking phrases from
    overriding explicit late/missing delivery signals.
    """

    text = normalize_message(
        message
    )


    # --------------------------------------------------------
    # Late / Missing Delivery
    # --------------------------------------------------------
    #
    # Explicit delay/missing language should take priority
    # over generic tracking language.
    # --------------------------------------------------------

    late_patterns = [

        r"\blate\b",

        r"\bdelayed\b",

        r"\bdelay\b",

        r"\bstill waiting\b",

        r"\bstill haven t received\b",

        r"\bstill have not received\b",

        r"\bnot arrived\b",

        r"\bhas not arrived\b",

        r"\bhasnt arrived\b",

        r"\bhasn't arrived\b",

        r"\bnever arrived\b",

        r"\bnot received\b",

        r"\bnot been received\b",

        r"\bmissing package\b",

        r"\bmissing order\b",

        r"\bmissing shipment\b",

        r"\bwhere is my package\b.*\bnot\b",

        r"\border is overdue\b",

        r"\bdelivery is overdue\b",

        r"\boverdue delivery\b",

        r"\bpast the delivery date\b",

        r"\bshould have arrived\b",

        r"\bsupposed to arrive\b.*\bbut\b"
    ]


    late_signal = any(
        re.search(
            pattern,
            text
        )
        for pattern in late_patterns
    )


    if late_signal:

        return (
            "Late / Missing Delivery",
            max(
                hybrid_confidence,
                0.75
            )
        )


    # --------------------------------------------------------
    # Order Tracking / Status
    # --------------------------------------------------------
    #
    # These are high-precision tracking/status expressions.
    # --------------------------------------------------------

    tracking_patterns = [

        r"\btrack my order\b",

        r"\btrack my package\b",

        r"\btrack my shipment\b",

        r"\btrack the order\b",

        r"\btrack the package\b",

        r"\btrack the shipment\b",

        r"\btracking number\b",

        r"\btracking link\b",

        r"\btracking information\b",

        r"\btracking info\b",

        r"\bwhere is my order\b",

        r"\bwhere is my package\b",

        r"\bwhere is my shipment\b",

        r"\bwhere s my order\b",

        r"\bwhere s my package\b",

        r"\bwhere s my shipment\b",

        r"\border status\b",

        r"\bpackage status\b",

        r"\bshipment status\b",

        r"\bcheck my order status\b",

        r"\bcheck the order status\b",

        r"\bcheck my package status\b",

        r"\bcheck shipment status\b",

        r"\bstatus of my order\b",

        r"\bstatus of my package\b",

        r"\bstatus of my shipment\b",

        r"\bhas my order shipped\b",

        r"\bhas my package shipped\b",

        r"\bhas my shipment shipped\b",

        r"\bwhere can i track\b",

        r"\bhow can i track\b",

        r"\bcan i track\b",

        r"\btrack an order\b",

        r"\btrack an item\b"
    ]


    tracking_signal = any(
        re.search(
            pattern,
            text
        )
        for pattern in tracking_patterns
    )


    if tracking_signal:

        return (
            "Order Tracking / Status",
            max(
                hybrid_confidence,
                0.75
            )
        )


    # --------------------------------------------------------
    # No guardrail triggered
    # --------------------------------------------------------

    return (
        hybrid_intent,
        hybrid_confidence
    )


# ============================================================
# 6. Hybrid Prediction
# ============================================================

def hybrid_predict(message):
    """
    Predict customer intent using:

    1. TF-IDF + Logistic Regression
    2. Semantic similarity
    3. Weighted hybrid scoring
    4. Conservative intent guardrails
    """

    message = str(message).strip()


    # --------------------------------------------------------
    # Empty message
    # --------------------------------------------------------

    if not message:

        return {
            "intent": "Unknown",

            "confidence": 0.0,

            "tfidf_intent": "Unknown",

            "tfidf_confidence": 0.0,

            "semantic_intent": "Unknown",

            "semantic_confidence": 0.0
        }


    # ========================================================
    # TF-IDF prediction
    # ========================================================

    tfidf_vector = tfidf.transform(
        [message]
    )


    tfidf_probabilities = (
        tfidf_model.predict_proba(
            tfidf_vector
        )[0]
    )


    tfidf_index = np.argmax(
        tfidf_probabilities
    )


    tfidf_intent = (
        tfidf_model.classes_[
            tfidf_index
        ]
    )


    tfidf_confidence = float(
        tfidf_probabilities[
            tfidf_index
        ]
    )


    # ========================================================
    # Semantic prediction
    # ========================================================

    message_embedding = semantic_model.encode(
        [message],
        normalize_embeddings=True
    )


    similarities = cosine_similarity(
        message_embedding,
        embeddings
    )[0]


    # Top 5 semantically similar examples
    top_indices = np.argsort(
        similarities
    )[::-1][:5]


    semantic_scores = {}


    for index in top_indices:

        intent = df.iloc[
            index
        ]["intent"]


        similarity = float(
            similarities[index]
        )


        if intent not in semantic_scores:

            semantic_scores[
                intent
            ] = 0.0


        semantic_scores[
            intent
        ] += similarity


    # --------------------------------------------------------
    # Best semantic intent
    # --------------------------------------------------------

    semantic_intent = max(
        semantic_scores,
        key=semantic_scores.get
    )


    semantic_confidence = float(
        similarities[
            top_indices[0]
        ]
    )


    # ========================================================
    # Hybrid scoring
    # ========================================================

    hybrid_scores = {}


    for label in labels:

        # ----------------------------------------------------
        # TF-IDF probability
        # ----------------------------------------------------

        class_index = list(
            tfidf_model.classes_
        ).index(label)


        tfidf_score = float(
            tfidf_probabilities[
                class_index
            ]
        )


        # ----------------------------------------------------
        # Semantic score
        # ----------------------------------------------------

        semantic_score = (
            semantic_scores.get(
                label,
                0.0
            )
        )


        # Normalize because up to five
        # semantic examples contribute.
        semantic_score = (
            semantic_score / 5.0
        )


        # ----------------------------------------------------
        # Weighted hybrid score
        # ----------------------------------------------------

        hybrid_scores[
            label
        ] = (
            0.55 * tfidf_score
            +
            0.45 * semantic_score
        )


    # --------------------------------------------------------
    # Raw hybrid prediction
    # --------------------------------------------------------

    raw_hybrid_intent = max(
        hybrid_scores,
        key=hybrid_scores.get
    )


    raw_confidence = (
        hybrid_scores[
            raw_hybrid_intent
        ]
    )


    raw_confidence = float(
        min(
            max(
                raw_confidence,
                0.0
            ),
            1.0
        )
    )


    # ========================================================
    # Apply guardrails
    # ========================================================

    final_intent, final_confidence = (
        apply_intent_guardrails(
            message=message,
            hybrid_intent=raw_hybrid_intent,
            hybrid_confidence=raw_confidence
        )
    )


    # ========================================================
    # Final result
    # ========================================================

    return {

        "intent":
            final_intent,

        "confidence":
            final_confidence,

        "tfidf_intent":
            tfidf_intent,

        "tfidf_confidence":
            tfidf_confidence,

        "semantic_intent":
            semantic_intent,

        "semantic_confidence":
            semantic_confidence
    }


# ============================================================
# 7. Test Examples
# ============================================================

def run_tests():
    """
    Run predefined examples to verify
    the classifier and guardrails.
    """

    test_messages = [

        "My order is delayed and has not arrived yet.",

        "I want to return this item and get my money back.",

        "The product I received is damaged.",

        "Can you tell me where my package is?",

        "I need help with my account.",

        "When will my order arrive?",

        "My device is not working.",

        "I need to contact Amazon customer support.",

        "Can I track my order?",

        "What is my tracking number?",

        "What is the status of my package?"
    ]


    print(
        "\n" + "=" * 65
    )

    print(
        "HYBRID MODEL TEST"
    )

    print(
        "=" * 65
    )


    for message in test_messages:

        result = hybrid_predict(
            message
        )


        print(
            "\nCustomer:"
        )

        print(
            message
        )


        print(
            "\nHybrid prediction:"
        )

        print(
            result["intent"]
        )


        print(
            f"Hybrid confidence: "
            f"{result['confidence']:.4f}"
        )


        print(
            f"TF-IDF prediction: "
            f"{result['tfidf_intent']} "
            f"({result['tfidf_confidence']:.4f})"
        )


        print(
            f"Semantic prediction: "
            f"{result['semantic_intent']} "
            f"({result['semantic_confidence']:.4f})"
        )


        print(
            "-" * 65
        )


# ============================================================
# 8. Interactive Mode
# ============================================================

def interactive_mode():
    """
    Interactive customer-support intent prediction.
    """

    print(
        "\n" + "=" * 65
    )

    print(
        "INTERACTIVE HYBRID MODE"
    )

    print(
        "Type 'exit' to stop."
    )

    print(
        "=" * 65
    )


    while True:

        try:

            message = input(
                "\nCustomer message: "
            ).strip()


        except KeyboardInterrupt:

            print(
                "\nExiting..."
            )

            break


        except EOFError:

            print(
                "\nExiting..."
            )

            break


        if message.lower() == "exit":

            print(
                "Goodbye!"
            )

            break


        if not message:

            continue


        result = hybrid_predict(
            message
        )


        print(
            "\nPredicted intent:"
        )

        print(
            result["intent"]
        )


        print(
            f"Hybrid confidence: "
            f"{result['confidence']:.4f}"
        )


        print(
            f"TF-IDF prediction: "
            f"{result['tfidf_intent']} "
            f"({result['tfidf_confidence']:.4f})"
        )


        print(
            f"Semantic prediction: "
            f"{result['semantic_intent']} "
            f"({result['semantic_confidence']:.4f})"
        )


        print(
            "\nDecision:"
        )


        if result["confidence"] >= 0.60:

            print(
                "AUTO-HANDLE"
            )

            print(
                "Reason: "
                "High-confidence intent prediction."
            )

        else:

            print(
                "ESCALATE"
            )

            print(
                "Reason: "
                "Low-confidence intent prediction; "
                "human verification recommended."
            )


        print(
            "-" * 65
        )


# ============================================================
# 9. Main Entry Point
# ============================================================

if __name__ == "__main__":

    run_tests()

    interactive_mode()
