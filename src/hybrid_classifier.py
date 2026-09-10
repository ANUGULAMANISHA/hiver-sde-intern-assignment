import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer


# ============================================================
# Hiver SDE Assignment - Hybrid Intent Classifier
#
# Hybrid approach:
#   1. Word + character TF-IDF + Logistic Regression
#   2. Semantic class-centroid similarity
#   3. Weighted combination
#
# IMPORTANT:
# The golden set is NEVER used for training.
# ============================================================


# ------------------------------------------------------------
# 1. Load historical training data
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "training_data.csv"

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        f"training_data.csv not found at: {TRAIN_FILE}\n"
        "Place training_data.csv in the project root."
    )


df = pd.read_csv(
    TRAIN_FILE
)


# The prepared dataset may use "text"
if (
    "customer_message" not in df.columns
    and "text" in df.columns
):

    df = df.rename(
        columns={
            "text": "customer_message"
        }
    )


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
        f"Missing required columns: {missing_columns}"
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


labels = sorted(
    df["intent"].unique()
)


print("=" * 70)
print("HIVER SDE ASSIGNMENT - HYBRID INTENT CLASSIFIER")
print("=" * 70)

print(
    f"\nTraining examples: {len(df)}"
)

print(
    f"Intents: {len(labels)}"
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
# 2. Word-level TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TRAINING WORD TF-IDF MODEL")
print("=" * 70)


word_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=40000,
    sublinear_tf=True
)


X_word = word_tfidf.fit_transform(
    df["customer_message"]
)


word_model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced"
)


word_model.fit(
    X_word,
    df["intent"]
)


print(
    "Word TF-IDF model trained successfully."
)


# ============================================================
# 3. Character-level TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CHARACTER TF-IDF MODEL")
print("=" * 70)


char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)


X_char = char_tfidf.fit_transform(
    df["customer_message"]
)


char_model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced"
)


char_model.fit(
    X_char,
    df["intent"]
)


print(
    "Character TF-IDF model trained successfully."
)


# ============================================================
# 4. Semantic model
# ============================================================

print("\n" + "=" * 70)
print("LOADING SEMANTIC MODEL")
print("=" * 70)


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


embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print(
    "Semantic embeddings created."
)


print(
    f"Embedding count: {len(embeddings)}"
)


# ============================================================
# 5. Create semantic class centroids
# ============================================================

print("\n" + "=" * 70)
print("CREATING SEMANTIC INTENT PROTOTYPES")
print("=" * 70)


class_centroids = {}


for label in labels:

    label_mask = (
        df["intent"].values
        == label
    )


    label_embeddings = (
        embeddings[label_mask]
    )


    centroid = (
        label_embeddings.mean(
            axis=0
        )
    )


    centroid = centroid / (
        np.linalg.norm(
            centroid
        )
        + 1e-12
    )


    class_centroids[label] = (
        centroid.astype("float32")
    )


print(
    f"Created {len(class_centroids)} "
    "semantic intent prototypes."
)


# ============================================================
# 6. Prediction helpers
# ============================================================

def _softmax(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    values = values - np.max(
        values
    )


    exp_values = np.exp(
        values
    )


    return (
        exp_values
        / (
            exp_values.sum()
            + 1e-12
        )
    )


# ------------------------------------------------------------
# Hybrid prediction
# ------------------------------------------------------------

def hybrid_predict(message):
    """
    Predict intent using three signals:

    1. Word TF-IDF probability
    2. Character TF-IDF probability
    3. Semantic similarity to intent centroids

    Final weights:

        40% word TF-IDF
        25% character TF-IDF
        35% semantic similarity
    """

    message = str(
        message
    ).strip()


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
    # Word TF-IDF
    # ========================================================

    word_vector = word_tfidf.transform(
        [message]
    )


    word_probabilities = (
        word_model.predict_proba(
            word_vector
        )[0]
    )


    word_classes = (
        word_model.classes_
    )


    word_best_index = int(
        np.argmax(
            word_probabilities
        )
    )


    word_intent = (
        word_classes[word_best_index]
    )


    word_confidence = float(
        word_probabilities[
            word_best_index
        ]
    )


    # ========================================================
    # Character TF-IDF
    # ========================================================

    char_vector = char_tfidf.transform(
        [message]
    )


    char_probabilities = (
        char_model.predict_proba(
            char_vector
        )[0]
    )


    char_classes = (
        char_model.classes_
    )


    # Make sure class ordering matches
    # the common label ordering.
    char_probability_map = {
        label: float(
            char_probabilities[
                list(char_classes).index(
                    label
                )
            ]
        )
        for label in labels
    }


    # ========================================================
    # Combined lexical probabilities
    # ========================================================

    word_probability_map = {
        label: float(
            word_probabilities[
                list(word_classes).index(
                    label
                )
            ]
        )
        for label in labels
    }


    lexical_scores = {}


    for label in labels:

        lexical_scores[label] = (
            0.60
            * word_probability_map[label]
            +
            0.40
            * char_probability_map[label]
        )


    # ========================================================
    # Semantic centroid similarity
    # ========================================================

    message_embedding = semantic_model.encode(
        [message],
        normalize_embeddings=True
    )[0]


    semantic_scores = {}


    for label in labels:

        similarity = float(
            np.dot(
                message_embedding,
                class_centroids[label]
            )
        )


        semantic_scores[label] = similarity


    # Convert cosine similarities into
    # normalized semantic probabilities.
    #
    # This prevents raw similarity values
    # from dominating the lexical model.

    semantic_values = np.array(
        [
            semantic_scores[label]
            for label in labels
        ],
        dtype=np.float64
    )


    semantic_probabilities = _softmax(
        semantic_values * 8.0
    )


    semantic_probability_map = {
        label: float(
            semantic_probabilities[index]
        )
        for index, label
        in enumerate(labels)
    }


    semantic_best_label = max(
        semantic_probability_map,
        key=semantic_probability_map.get
    )


    semantic_confidence = (
        semantic_probability_map[
            semantic_best_label
        ]
    )


    # ========================================================
    # Final hybrid score
    # ========================================================

    hybrid_scores = {}


    for label in labels:

        hybrid_scores[label] = (

            0.65
            * lexical_scores[label]

            +

            0.35
            * semantic_probability_map[label]
        )


    hybrid_intent = max(
        hybrid_scores,
        key=hybrid_scores.get
    )


    # Normalize final hybrid scores
    hybrid_values = np.array(
        [
            hybrid_scores[label]
            for label in labels
        ],
        dtype=np.float64
    )


    hybrid_probabilities = _softmax(
        hybrid_values * 10.0
    )


    hybrid_probability_map = {
        label: float(
            hybrid_probabilities[index]
        )
        for index, label
        in enumerate(labels)
    }


    hybrid_confidence = (
        hybrid_probability_map[
            hybrid_intent
        ]
    )


    # ========================================================
    # Return prediction
    # ========================================================

    return {

        "intent":
            hybrid_intent,

        "confidence":
            float(
                hybrid_confidence
            ),

        "tfidf_intent":
            word_intent,

        "tfidf_confidence":
            float(
                word_confidence
            ),

        "semantic_intent":
            semantic_best_label,

        "semantic_confidence":
            float(
                semantic_confidence
            )
    }


# ============================================================
# 7. Test examples
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

        "I need to contact Amazon customer support."
    ]


    print("\n" + "=" * 70)
    print("HYBRID MODEL TEST")
    print("=" * 70)


    for message in test_messages:

        result = hybrid_predict(
            message
        )


        print("\nCustomer:")
        print(
            message
        )


        print("\nHybrid prediction:")
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
            "-" * 70
        )


# ============================================================
# 8. Interactive mode
# ============================================================

def interactive_mode():

    print("\n" + "=" * 70)
    print("INTERACTIVE HYBRID MODE")
    print("Type 'exit' to stop.")
    print("=" * 70)


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
            "-" * 70
        )


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    run_tests()

    interactive_mode()
