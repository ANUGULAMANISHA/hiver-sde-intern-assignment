import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


# ============================================================
# Hiver SDE Assignment - Hybrid Intent Classifier
# ============================================================
#
# Combines:
#   1. TF-IDF + Logistic Regression
#   2. Semantic similarity using all-MiniLM-L6-v2
#   3. Delivery/tracking guardrails
#
# The classifier exposes:
#   predict_intent()
#   predict_with_details()
#   hybrid_predict()
#
# support_agent.py expects the detailed prediction fields:
#   tfidf_intent
#   tfidf_confidence
#   semantic_intent
#   semantic_confidence
#   hybrid_intent
#   hybrid_confidence
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "training_data.csv"
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"training_data.csv not found at: {TRAIN_FILE}\n"
        "Place training_data.csv in the project root."
    )


df = pd.read_csv(
    TRAIN_FILE
)


# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================

# Some versions of the prepared dataset use "text"
# while the agent expects "customer_message".

if (
    "customer_message" not in df.columns
    and "text" in df.columns
):

    df = df.rename(
        columns={
            "text": "customer_message"
        }
    )


# ============================================================
# 4. VALIDATE COLUMNS
# ============================================================

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


# ============================================================
# 5. CLEAN TRAINING DATA
# ============================================================

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


# ============================================================
# 6. DISPLAY TRAINING INFORMATION
# ============================================================

print("=" * 65)

print(
    "HIVER SDE ASSIGNMENT - HYBRID INTENT CLASSIFIER"
)

print("=" * 65)


print(
    f"\nTraining examples: {len(df)}"
)


print(
    f"Number of intents: {df['intent'].nunique()}"
)


print(
    "\nIntents:"
)


for label in sorted(
    df["intent"].unique()
):

    print(
        f"- {label}"
    )


print(
    "\nIntent distribution:"
)


print(
    df["intent"].value_counts()
)


# ============================================================
# 7. TF-IDF FEATURE EXTRACTION
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "CREATING TF-IDF FEATURES"
)

print(
    "=" * 65
)


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(
        1,
        2
    ),

    min_df=2,

    max_features=30000,

    sublinear_tf=True
)


X = vectorizer.fit_transform(
    df["customer_message"]
)


print(
    "TF-IDF features created successfully."
)


print(
    f"Feature matrix shape: {X.shape}"
)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "TRAINING LOGISTIC REGRESSION"
)

print(
    "=" * 65
)


model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced"
)


model.fit(
    X,
    df["intent"]
)


print(
    "Logistic Regression model trained successfully."
)


# ============================================================
# 9. LOAD SEMANTIC MODEL
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
    "Semantic model loaded successfully."
)


# ============================================================
# 10. CREATE TRAINING EMBEDDINGS
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "CREATING SEMANTIC EMBEDDINGS"
)

print(
    "=" * 65
)


training_messages = (
    df["customer_message"]
    .astype(str)
    .tolist()
)


training_embeddings = semantic_model.encode(

    training_messages,

    normalize_embeddings=True,

    show_progress_bar=True
)


training_embeddings = np.asarray(
    training_embeddings,
    dtype=np.float32
)


print(
    "Training embeddings created successfully."
)


print(
    f"Embedding matrix shape: "
    f"{training_embeddings.shape}"
)


# ============================================================
# 11. NORMALIZE TEXT
# ============================================================

def normalize_text(message):

    message = str(
        message
    ).lower()


    message = re.sub(
        r"[^a-z0-9\s]",
        " ",
        message
    )


    message = re.sub(
        r"\s+",
        " ",
        message
    )


    return message.strip()


# ============================================================
# 12. DELIVERY GUARDRAILS
# ============================================================

def delivery_guardrail(message):

    text = normalize_text(
        message
    )


    # --------------------------------------------------------
    # Late / Missing Delivery
    # --------------------------------------------------------

    late_patterns = [

        "delayed",

        "delay",

        "late",

        "overdue",

        "still waiting",

        "not arrived",

        "hasnt arrived",

        "has not arrived",

        "didnt arrive",

        "did not arrive",

        "not received",

        "hasnt been delivered",

        "has not been delivered",

        "missing package",

        "missing order",

        "missing shipment",

        "package is missing",

        "order is missing",

        "shipment is missing",

        "past delivery date",

        "past the delivery date",

        "should have arrived",

        "supposed to arrive",

        "supposed to be here",

        "delivery was yesterday",

        "delivery date passed",

        "delivery date has passed"
    ]


    for pattern in late_patterns:

        if pattern in text:

            return (
                "Late / Missing Delivery"
            )


    # --------------------------------------------------------
    # Order Tracking / Status
    # --------------------------------------------------------

    tracking_patterns = [

        "track my order",

        "track my package",

        "track my shipment",

        "tracking number",

        "tracking link",

        "tracking information",

        "tracking info",

        "where is my order",

        "where is my package",

        "where is my shipment",

        "order status",

        "package status",

        "shipment status",

        "status of my order",

        "status of my package",

        "status of my shipment",

        "check my order status",

        "check order status",

        "has my order shipped",

        "has my package shipped",

        "has my shipment shipped",

        "can i track my order",

        "can i track my package",

        "can i track an order",

        "how can i track my order",

        "how can i track my package",

        "where can i track my order",

        "where can i track my package",

        "track an order",

        "track my item"
    ]


    for pattern in tracking_patterns:

        if pattern in text:

            return (
                "Order Tracking / Status"
            )


    return None


# ============================================================
# 13. SEMANTIC INTENT SCORES
# ============================================================

def semantic_intent_scores(message):

    message_embedding = (
        semantic_model.encode(
            [str(message)],
            normalize_embeddings=True
        )
    )


    message_embedding = np.asarray(
        message_embedding,
        dtype=np.float32
    )[0]


    # Because both vectors are normalized,
    # dot product = cosine similarity.

    similarities = np.dot(
        training_embeddings,
        message_embedding
    )


    semantic_scores = {}


    for intent in model.classes_:

        intent_indices = np.where(
            df["intent"].values == intent
        )[0]


        if len(intent_indices) == 0:

            semantic_scores[intent] = 0.0

            continue


        intent_similarities = (
            similarities[
                intent_indices
            ]
        )


        top_k = min(
            5,
            len(intent_similarities)
        )


        top_scores = np.sort(
            intent_similarities
        )[-top_k:]


        semantic_scores[intent] = float(
            np.mean(top_scores)
        )


    return semantic_scores


# ============================================================
# 14. SIMPLE TF-IDF PREDICTION
# ============================================================

def predict_intent(message):

    message = str(
        message
    ).strip()


    if not message:

        return {
            "intent": "Unknown",
            "confidence": 0.0
        }


    message_vector = (
        vectorizer.transform(
            [message]
        )
    )


    probabilities = (
        model.predict_proba(
            message_vector
        )[0]
    )


    best_index = (
        probabilities.argmax()
    )


    predicted_intent = (
        model.classes_[
            best_index
        ]
    )


    confidence = float(
        probabilities[
            best_index
        ]
    )


    return {

        "intent":
            predicted_intent,

        "confidence":
            confidence
    }


# ============================================================
# 15. FULL HYBRID PREDICTION
# ============================================================

def predict_with_details(message):

    """
    Perform:

        TF-IDF prediction
        +
        semantic prediction
        +
        hybrid prediction
        +
        delivery guardrails

    Returns all fields required by support_agent.py.
    """

    message = str(
        message
    ).strip()


    # --------------------------------------------------------
    # Empty message
    # --------------------------------------------------------

    if not message:

        return {

            "intent":
                "Unknown",

            "confidence":
                0.0,

            "tfidf_intent":
                "Unknown",

            "tfidf_confidence":
                0.0,

            "semantic_intent":
                "Unknown",

            "semantic_confidence":
                0.0,

            "hybrid_intent":
                "Unknown",

            "hybrid_confidence":
                0.0,

            "probabilities":
                {},

            "semantic_scores":
                {},

            "hybrid_scores":
                {}
        }


    # ========================================================
    # TF-IDF PREDICTION
    # ========================================================

    message_vector = (
        vectorizer.transform(
            [message]
        )
    )


    probabilities = (
        model.predict_proba(
            message_vector
        )[0]
    )


    best_tfidf_index = (
        probabilities.argmax()
    )


    tfidf_intent = (
        model.classes_[
            best_tfidf_index
        ]
    )


    tfidf_confidence = float(
        probabilities[
            best_tfidf_index
        ]
    )


    probability_dict = {

        label: float(
            probability
        )

        for label, probability
        in zip(
            model.classes_,
            probabilities
        )
    }


    # ========================================================
    # SEMANTIC PREDICTION
    # ========================================================

    semantic_scores = (
        semantic_intent_scores(
            message
        )
    )


    semantic_intent = max(
        semantic_scores,
        key=semantic_scores.get
    )


    semantic_confidence = float(
        semantic_scores[
            semantic_intent
        ]
    )


    # ========================================================
    # HYBRID SCORE
    # ========================================================
    #
    # 55% TF-IDF
    # 45% semantic signal
    #
    # The semantic similarity is scaled before combining
    # with the TF-IDF probability.
    # ========================================================

    hybrid_scores = {}


    for intent in model.classes_:

        tfidf_score = (
            probability_dict.get(
                intent,
                0.0
            )
        )


        semantic_score = (
            semantic_scores.get(
                intent,
                0.0
            )
        )


        hybrid_scores[intent] = (

            0.55 * tfidf_score

            +

            0.45 * (
                semantic_score / 5.0
            )
        )


    hybrid_intent = max(
        hybrid_scores,
        key=hybrid_scores.get
    )


    hybrid_confidence = float(
        hybrid_scores[
            hybrid_intent
        ]
    )


    # ========================================================
    # DELIVERY GUARDRAIL
    # ========================================================

    guardrail_intent = (
        delivery_guardrail(
            message
        )
    )


    if guardrail_intent is not None:

        hybrid_intent = (
            guardrail_intent
        )


        hybrid_confidence = max(

            hybrid_confidence,

            probability_dict.get(
                guardrail_intent,
                0.0
            )
        )


    # ========================================================
    # RETURN COMPLETE RESULT
    # ========================================================

    return {

        # Final hybrid prediction
        "intent":
            hybrid_intent,

        "confidence":
            hybrid_confidence,


        # TF-IDF result
        "tfidf_intent":
            tfidf_intent,

        "tfidf_confidence":
            tfidf_confidence,


        # Semantic result
        "semantic_intent":
            semantic_intent,

        "semantic_confidence":
            semantic_confidence,


        # Hybrid result
        "hybrid_intent":
            hybrid_intent,

        "hybrid_confidence":
            hybrid_confidence,


        # Detailed scores
        "probabilities":
            probability_dict,

        "semantic_scores":
            semantic_scores,

        "hybrid_scores":
            hybrid_scores
    }


# ============================================================
# 16. COMPATIBILITY FUNCTION
# ============================================================

def hybrid_predict(message):

    """
    Compatibility wrapper for support_agent.py.

    support_agent.py imports:

        from src.hybrid_classifier import hybrid_predict

    """

    return predict_with_details(
        message
    )


# ============================================================
# 17. TEST EXAMPLES
# ============================================================

def run_tests():

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

        result = (
            predict_with_details(
                message
            )
        )


        print(
            "\nCustomer:"
        )

        print(
            message
        )


        print(
            "\nTF-IDF intent:"
        )

        print(
            result[
                "tfidf_intent"
            ]
        )


        print(
            "Semantic intent:"
        )

        print(
            result[
                "semantic_intent"
            ]
        )


        print(
            "Hybrid intent:"
        )

        print(
            result[
                "hybrid_intent"
            ]
        )


        print(
            f"Hybrid confidence: "
            f"{result['hybrid_confidence']:.4f}"
        )


        print(
            "-" * 65
        )


# ============================================================
# 18. INTERACTIVE MODE
# ============================================================

def interactive_mode():

    print(
        "\n" + "=" * 65
    )

    print(
        "INTERACTIVE HYBRID CLASSIFIER"
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


        except (
            KeyboardInterrupt,
            EOFError
        ):

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


        result = (
            predict_with_details(
                message
            )
        )


        print(
            "\nPredicted intent:"
        )

        print(
            result[
                "intent"
            ]
        )


        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )


        print(
            "\nTF-IDF intent:"
        )

        print(
            result[
                "tfidf_intent"
            ]
        )


        print(
            "\nSemantic intent:"
        )

        print(
            result[
                "semantic_intent"
            ]
        )


        print(
            "\nHybrid intent:"
        )

        print(
            result[
                "hybrid_intent"
            ]
        )


        print(
            "-" * 65
        )


# ============================================================
# 19. MAIN
# ============================================================

if __name__ == "__main__":

    run_tests()

    interactive_mode()