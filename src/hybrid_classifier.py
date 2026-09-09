import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Hiver SDE Assignment - Hybrid Intent Classifier
# ============================================================

print("=" * 65)
print("HIVER SDE ASSIGNMENT - HYBRID INTENT CLASSIFIER")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load historical training data
# ------------------------------------------------------------

TRAIN_FILE = "training_data.csv"

df = pd.read_csv(TRAIN_FILE)

# The prepared training file uses "text".
# Rename it to "customer_message" for consistency.
if "customer_message" not in df.columns and "text" in df.columns:
    df = df.rename(columns={"text": "customer_message"})

# Check required columns
required_columns = ["customer_message", "intent"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

# Remove empty rows
df = df.dropna(
    subset=["customer_message", "intent"]
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
    (df["customer_message"] != "") &
    (df["intent"] != "")
].copy()


print(f"\nTraining examples: {len(df)}")
print(f"Intents: {df['intent'].nunique()}")

labels = sorted(
    df["intent"].unique()
)

print("\nIntents:")

for label in labels:
    print(f"- {label}")

print("\nIntent distribution:")
print(df["intent"].value_counts())


# ============================================================
# 2. TF-IDF + Logistic Regression
# ============================================================

print("\n" + "=" * 65)
print("TRAINING TF-IDF + LOGISTIC REGRESSION")
print("=" * 65)

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

print("TF-IDF model trained successfully.")


# ============================================================
# 3. Semantic Embeddings
# ============================================================

print("\n" + "=" * 65)
print("LOADING SEMANTIC MODEL")
print("=" * 65)

semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating semantic embeddings...")

embeddings = semantic_model.encode(
    df["customer_message"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Semantic embeddings created.")

print(f"Embedding count: {len(embeddings)}")


# ============================================================
# 4. Hybrid Prediction
# ============================================================

def hybrid_predict(message):
    """
    Predict customer intent using:

    1. TF-IDF + Logistic Regression
    2. Semantic similarity
    3. Weighted hybrid scoring
    """

    message = str(message).strip()

    if not message:
        return {
            "intent": "Unknown",
            "confidence": 0.0,
            "tfidf_intent": "Unknown",
            "tfidf_confidence": 0.0,
            "semantic_intent": "Unknown",
            "semantic_confidence": 0.0
        }

    # --------------------------------------------------------
    # TF-IDF prediction
    # --------------------------------------------------------

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
        tfidf_model.classes_[tfidf_index]
    )

    tfidf_confidence = float(
        tfidf_probabilities[tfidf_index]
    )


    # --------------------------------------------------------
    # Semantic prediction
    # --------------------------------------------------------

    message_embedding = semantic_model.encode(
        [message],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        message_embedding,
        embeddings
    )[0]

    # Get top 5 similar historical examples
    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    semantic_scores = {}

    for index in top_indices:

        intent = df.iloc[index]["intent"]

        similarity = float(
            similarities[index]
        )

        if intent not in semantic_scores:
            semantic_scores[intent] = 0.0

        semantic_scores[intent] += similarity


    # Best semantic intent
    semantic_intent = max(
        semantic_scores,
        key=semantic_scores.get
    )

    semantic_confidence = float(
        similarities[top_indices[0]]
    )


    # --------------------------------------------------------
    # Hybrid scoring
    # --------------------------------------------------------

    hybrid_scores = {}

    for label in labels:

        # TF-IDF probability
        class_index = list(
            tfidf_model.classes_
        ).index(label)

        tfidf_score = float(
            tfidf_probabilities[class_index]
        )

        # Semantic similarity vote
        semantic_score = semantic_scores.get(
            label,
            0.0
        )

        # Normalize semantic score because
        # up to 5 examples contribute to it
        semantic_score = semantic_score / 5.0

        # Weighted combination
        hybrid_scores[label] = (
            0.55 * tfidf_score +
            0.45 * semantic_score
        )


    # Final prediction
    hybrid_intent = max(
        hybrid_scores,
        key=hybrid_scores.get
    )

    raw_confidence = hybrid_scores[
        hybrid_intent
    ]

    # Keep confidence in 0-1 range
    hybrid_confidence = float(
        min(max(raw_confidence, 0.0), 1.0)
    )


    return {
        "intent": hybrid_intent,
        "confidence": hybrid_confidence,
        "tfidf_intent": tfidf_intent,
        "tfidf_confidence": tfidf_confidence,
        "semantic_intent": semantic_intent,
        "semantic_confidence": semantic_confidence
    }


# ============================================================
# 5. Test Examples
# ============================================================

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


print("\n" + "=" * 65)
print("HYBRID MODEL TEST")
print("=" * 65)


for message in test_messages:

    result = hybrid_predict(
        message
    )

    print("\nCustomer:")
    print(message)

    print("\nHybrid prediction:")
    print(result["intent"])

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

    print("-" * 65)


# ============================================================
# 6. Interactive Mode
# ============================================================

print("\n" + "=" * 65)
print("INTERACTIVE HYBRID MODE")
print("Type 'exit' to stop.")
print("=" * 65)


while True:

    try:

        message = input(
            "\nCustomer message: "
        ).strip()

    except KeyboardInterrupt:

        print("\nExiting...")
        break

    except EOFError:

        print("\nExiting...")
        break


    if message.lower() == "exit":

        print("Goodbye!")
        break


    if not message:

        continue


    result = hybrid_predict(
        message
    )


    print("\nPredicted intent:")
    print(result["intent"])

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

    # Simple confidence-based decision
    if result["confidence"] >= 0.60:

        print("AUTO-HANDLE")

        print(
            "Reason: "
            "High-confidence intent prediction."
        )

    else:

        print("ESCALATE")

        print(
            "Reason: "
            "Low-confidence intent prediction; "
            "human verification recommended."
        )

    print("-" * 65)