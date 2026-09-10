import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Hiver SDE Assignment - TF-IDF Intent Classifier
# ============================================================
#
# This classifier is a simple supervised baseline.
#
# IMPORTANT:
# - training_data.csv is used for training
# - golden_set.csv is NOT used for training
# - The golden set is reserved for evaluation
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "training_data.csv"


# ------------------------------------------------------------
# 2. Load training data
# ------------------------------------------------------------

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        f"training_data.csv not found at: {TRAIN_FILE}\n"
        "Place training_data.csv in the project root directory."
    )


df = pd.read_csv(
    TRAIN_FILE
)


# ------------------------------------------------------------
# 3. Normalize column names
# ------------------------------------------------------------

# The prepared training file normally contains "text".
# Convert it to "customer_message" for consistency.

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
# 4. Validate columns
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
        f"Missing required columns: "
        f"{missing_columns}\n"
        f"Available columns: "
        f"{list(df.columns)}"
    )


# ------------------------------------------------------------
# 5. Clean training data
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


# Remove empty rows

df = df[
    (df["customer_message"] != "")
    &
    (df["intent"] != "")
].copy()


# ------------------------------------------------------------
# 6. Display training information
# ------------------------------------------------------------

print("=" * 65)

print(
    "HIVER SDE ASSIGNMENT - TF-IDF INTENT CLASSIFIER"
)

print("=" * 65)


print(
    f"\nTraining examples: "
    f"{len(df)}"
)


print(
    f"Number of intents: "
    f"{df['intent'].nunique()}"
)


labels = sorted(
    df["intent"].unique()
)


print(
    "\nIntents:"
)


for label in labels:

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
# 7. TF-IDF feature extraction
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
    f"Feature matrix shape: "
    f"{X.shape}"
)


# ============================================================
# 8. Logistic Regression model
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
# 9. Prediction function
# ============================================================

def predict_intent(message):
    """
    Predict the intent of a customer message.

    Returns:

    {
        "intent": predicted intent,
        "confidence": probability of predicted intent
    }
    """

    message = str(
        message
    ).strip()


    # --------------------------------------------------------
    # Empty message
    # --------------------------------------------------------

    if not message:

        return {
            "intent": "Unknown",
            "confidence": 0.0
        }


    # --------------------------------------------------------
    # Transform input using the fitted TF-IDF vectorizer
    # --------------------------------------------------------

    message_vector = vectorizer.transform(
        [message]
    )


    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        message_vector
    )[0]


    # Find highest probability

    best_index = probabilities.argmax()


    predicted_intent = (
        model.classes_[best_index]
    )


    confidence = float(
        probabilities[best_index]
    )


    return {

        "intent":
            predicted_intent,

        "confidence":
            confidence
    }


# ============================================================
# 10. Detailed prediction
# ============================================================

def predict_with_details(message):
    """
    Return the predicted intent together with
    the probability assigned to every intent.
    """

    message = str(
        message
    ).strip()


    if not message:

        return {
            "intent": "Unknown",
            "confidence": 0.0,
            "probabilities": {}
        }


    message_vector = vectorizer.transform(
        [message]
    )


    probabilities = model.predict_proba(
        message_vector
    )[0]


    best_index = probabilities.argmax()


    predicted_intent = (
        model.classes_[best_index]
    )


    confidence = float(
        probabilities[best_index]
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


    return {

        "intent":
            predicted_intent,

        "confidence":
            confidence,

        "probabilities":
            probability_dict
    }


# ============================================================
# 11. Test examples
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
        "TF-IDF MODEL TEST"
    )

    print(
        "=" * 65
    )


    for message in test_messages:

        result = predict_intent(
            message
        )


        print(
            "\nCustomer:"
        )

        print(
            message
        )


        print(
            "\nPredicted intent:"
        )

        print(
            result["intent"]
        )


        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )


        print(
            "-" * 65
        )


# ============================================================
# 12. Interactive mode
# ============================================================

def interactive_mode():

    print(
        "\n" + "=" * 65
    )

    print(
        "INTERACTIVE TF-IDF CLASSIFIER"
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


        result = predict_with_details(
            message
        )


        print(
            "\nPredicted intent:"
        )

        print(
            result["intent"]
        )


        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )


        print(
            "\nAll intent probabilities:"
        )


        sorted_probabilities = sorted(

            result["probabilities"].items(),

            key=lambda item: item[1],

            reverse=True
        )


        for intent, probability in (
            sorted_probabilities
        ):

            print(
                f"{intent}: "
                f"{probability:.4f}"
            )


        print(
            "-" * 65
        )


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    run_tests()

    interactive_mode()
