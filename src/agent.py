import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Hiver SDE Assignment
# AI Customer Support Agent
# ============================================================

print("=" * 70)
print("AMAZONHELP AI CUSTOMER SUPPORT AGENT")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

TRAIN_FILE = "training_data.csv"

df = pd.read_csv(TRAIN_FILE)

# Handle either "text" or "customer_message"
if "customer_message" not in df.columns and "text" in df.columns:
    df = df.rename(columns={"text": "customer_message"})

required = ["customer_message", "intent"]

missing = [
    column for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

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

print(f"\nHistorical training examples: {len(df)}")
print(f"Number of intents: {df['intent'].nunique()}")


# ============================================================
# 2. TF-IDF CLASSIFIER
# ============================================================

print("\nTraining TF-IDF classifier...")

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)

X = tfidf.fit_transform(
    df["customer_message"]
)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X,
    df["intent"]
)

print("TF-IDF classifier ready.")


# ============================================================
# 3. SEMANTIC MODEL
# ============================================================

print("\nLoading semantic model...")

semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating historical embeddings...")

historical_embeddings = semantic_model.encode(
    df["customer_message"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(
    f"Created {len(historical_embeddings)} historical embeddings."
)


# ============================================================
# 4. RETRIEVE HISTORICAL CASES
# ============================================================

def retrieve_cases(message, top_k=5):

    query_embedding = semantic_model.encode(
        [message],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    cases = []

    for index in top_indices:

        cases.append({
            "customer_message": df.iloc[index][
                "customer_message"
            ],
            "response": df.iloc[index][
                "response"
            ] if "response" in df.columns else "",
            "intent": df.iloc[index]["intent"],
            "similarity": float(
                similarities[index]
            )
        })

    return cases


# ============================================================
# 5. HYBRID INTENT CLASSIFICATION
# ============================================================

def classify_intent(message):

    # -----------------------------
    # TF-IDF prediction
    # -----------------------------

    tfidf_vector = tfidf.transform(
        [message]
    )

    probabilities = classifier.predict_proba(
        tfidf_vector
    )[0]

    best_index = np.argmax(
        probabilities
    )

    tfidf_intent = classifier.classes_[
        best_index
    ]

    tfidf_confidence = float(
        probabilities[best_index]
    )


    # -----------------------------
    # Semantic prediction
    # -----------------------------

    cases = retrieve_cases(
        message,
        top_k=5
    )

    semantic_votes = {}

    for case in cases:

        intent = case["intent"]
        similarity = case["similarity"]

        semantic_votes[intent] = (
            semantic_votes.get(intent, 0)
            + similarity
        )

    semantic_intent = max(
        semantic_votes,
        key=semantic_votes.get
    )

    semantic_confidence = cases[0][
        "similarity"
    ]


    # -----------------------------
    # Hybrid decision
    # -----------------------------

    if tfidf_intent == semantic_intent:

        final_intent = tfidf_intent

        final_confidence = (
            0.60 * tfidf_confidence
            + 0.40 * semantic_confidence
        )

    else:

        # Prefer the model with stronger evidence
        if tfidf_confidence >= semantic_confidence:

            final_intent = tfidf_intent

            final_confidence = (
                0.65 * tfidf_confidence
                + 0.35 * semantic_confidence
            )

        else:

            final_intent = semantic_intent

            final_confidence = (
                0.35 * tfidf_confidence
                + 0.65 * semantic_confidence
            )


    final_confidence = float(
        min(max(final_confidence, 0.0), 1.0)
    )

    return (
        final_intent,
        final_confidence,
        tfidf_intent,
        tfidf_confidence,
        semantic_intent,
        semantic_confidence,
        cases
    )


# ============================================================
# 6. ESCALATION POLICY
# ============================================================

def decide_escalation(
    intent,
    confidence,
    message,
    cases
):

    message_lower = message.lower()

    # ----------------------------------------
    # High-risk / sensitive situations
    # ----------------------------------------

    sensitive_keywords = [
        "password",
        "account hacked",
        "hacked",
        "credit card",
        "bank account",
        "fraud",
        "stolen",
        "security",
        "unauthorized payment",
        "charge dispute"
    ]

    for keyword in sensitive_keywords:

        if keyword in message_lower:

            return (
                "ESCALATE",
                f"Sensitive issue detected: {keyword}"
            )


    # ----------------------------------------
    # Very low confidence
    # ----------------------------------------

    if confidence < 0.50:

        return (
            "ESCALATE",
            "Low intent confidence; human verification required."
        )


    # ----------------------------------------
    # Weak historical evidence
    # ----------------------------------------

    if not cases:

        return (
            "ESCALATE",
            "No relevant historical support case found."
        )


    best_similarity = cases[0]["similarity"]

    if best_similarity < 0.55:

        return (
            "ESCALATE",
            "Historical evidence is not sufficiently similar."
        )


    # ----------------------------------------
    # Account-specific actions
    # ----------------------------------------

    account_specific = [
        "change my account",
        "delete my account",
        "close my account",
        "update my payment",
        "change my payment"
    ]

    for keyword in account_specific:

        if keyword in message_lower:

            return (
                "ESCALATE",
                "Account-specific action requires human verification."
            )


    # ----------------------------------------
    # Otherwise auto-handle
    # ----------------------------------------

    return (
        "AUTO-HANDLE",
        "High-confidence intent with relevant historical evidence."
    )


# ============================================================
# 7. GENERATE GROUNDED REPLY
# ============================================================

def generate_reply(
    message,
    intent,
    decision,
    cases
):

    # ----------------------------------------
    # Escalation response
    # ----------------------------------------

    if decision == "ESCALATE":

        return (
            "Thanks for reaching out. "
            "We'd like to take a closer look at this and "
            "make sure we provide the right assistance. "
            "Please contact our support team so we can "
            "review the details with you."
        )


    # ----------------------------------------
    # No evidence
    # ----------------------------------------

    if not cases:

        return (
            "Thanks for contacting AmazonHelp. "
            "We'd be happy to help. Please share a few "
            "more details so we can assist you."
        )


    # ----------------------------------------
    # Use strongest historical response
    # ----------------------------------------

    best_case = cases[0]

    historical_response = (
        best_case.get("response", "")
    )

    if historical_response:

        # Remove Twitter agent ID where possible
        response = historical_response.strip()

        return response


    # ----------------------------------------
    # Fallback intent-based responses
    # ----------------------------------------

    responses = {

        "Order Tracking / Status":
            "We're happy to help with your order status. "
            "Please share the relevant order details so we "
            "can look into this for you.",

        "Late / Missing Delivery":
            "I'm sorry for the delay. "
            "Please contact us with your order details so "
            "we can look into the delivery status for you.",

        "Delivery Date / Shipping":
            "We'd be happy to help with your delivery details. "
            "Please share the relevant order information so "
            "we can assist you.",

        "Return / Refund":
            "We'd be happy to help with your return or refund. "
            "Please share the relevant order details so we "
            "can look into this for you.",

        "Damaged / Defective Product":
            "I'm sorry to hear that the product arrived damaged. "
            "Please contact us with the order details so we "
            "can look into this for you.",

        "Product / Device Support":
            "We'd be happy to help troubleshoot the product. "
            "Please share more details about the issue so "
            "we can assist you.",

        "Account / Contact Support":
            "We'd be happy to help. "
            "Please contact our support team with the relevant "
            "details so we can assist you."
    }

    return responses.get(
        intent,
        "Thanks for contacting AmazonHelp. "
        "Please share more details so we can assist you."
    )


# ============================================================
# 8. COMPLETE AGENT
# ============================================================

def run_agent(message):

    (
        intent,
        confidence,
        tfidf_intent,
        tfidf_confidence,
        semantic_intent,
        semantic_confidence,
        cases
    ) = classify_intent(message)


    decision, reason = decide_escalation(
        intent,
        confidence,
        message,
        cases
    )


    reply = generate_reply(
        message,
        intent,
        decision,
        cases
    )


    return {
        "intent": intent,
        "confidence": confidence,
        "tfidf_intent": tfidf_intent,
        "tfidf_confidence": tfidf_confidence,
        "semantic_intent": semantic_intent,
        "semantic_confidence": semantic_confidence,
        "decision": decision,
        "reason": reason,
        "reply": reply,
        "evidence": cases
    }


# ============================================================
# 9. TEST THE AGENT
# ============================================================

test_messages = [

    "My package is late and I still haven't received it.",

    "I want to return this item and get my money back.",

    "The product I received is damaged.",

    "Can you tell me where my package is?",

    "I need help with my account.",

    "My device is not working."
]


print("\n" + "=" * 70)
print("AI SUPPORT AGENT TEST")
print("=" * 70)


for message in test_messages:

    result = run_agent(message)

    print("\nCustomer:")
    print(message)

    print("\nPredicted intent:")
    print(result["intent"])

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print("\nTF-IDF:")
    print(
        f"{result['tfidf_intent']} "
        f"({result['tfidf_confidence']:.4f})"
    )

    print("\nSemantic:")
    print(
        f"{result['semantic_intent']} "
        f"({result['semantic_confidence']:.4f})"
    )

    print("\nDecision:")
    print(result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("\nDraft reply:")
    print(result["reply"])

    print("\nHistorical evidence:")

    for i, case in enumerate(
        result["evidence"][:3],
        start=1
    ):

        print(
            f"\nCase {i} "
            f"(similarity={case['similarity']:.4f})"
        )

        print(
            "Customer:",
            case["customer_message"]
        )

        print(
            "Response:",
            case["response"]
        )

    print("\n" + "-" * 70)


# ============================================================
# 10. INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE AI SUPPORT AGENT")
print("Type 'exit' to stop.")
print("=" * 70)


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


    result = run_agent(message)


    print("\n" + "=" * 50)

    print("INTENT:")
    print(result["intent"])

    print(
        f"\nCONFIDENCE: "
        f"{result['confidence']:.4f}"
    )

    print("\nDECISION:")
    print(result["decision"])

    print("\nREASON:")
    print(result["reason"])

    print("\nDRAFT REPLY:")
    print(result["reply"])

    print("\nTOP HISTORICAL EVIDENCE:")

    for i, case in enumerate(
        result["evidence"][:3],
        start=1
    ):

        print(
            f"\n{i}. Similarity: "
            f"{case['similarity']:.4f}"
        )

        print(
            "Customer:",
            case["customer_message"]
        )

        print(
            "Historical response:",
            case["response"]
        )

    print("=" * 50)