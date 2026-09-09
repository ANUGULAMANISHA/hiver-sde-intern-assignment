import pandas as pd
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# Hiver SDE Assignment - Fast Historical Retrieval
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RETRIEVAL_FILE = PROJECT_ROOT / "amazonhelp_retrieval_data.csv"
INDEX_FILE = PROJECT_ROOT / "amazonhelp.index"


# ------------------------------------------------------------
# Load retrieval data
# ------------------------------------------------------------

if not RETRIEVAL_FILE.exists():
    raise FileNotFoundError(
        f"Retrieval data not found: {RETRIEVAL_FILE}"
    )

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_FILE}"
    )


df = pd.read_csv(RETRIEVAL_FILE)

index = faiss.read_index(
    str(INDEX_FILE)
)


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

required_columns = [
    "customer_message",
    "text"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ------------------------------------------------------------
# Retrieval function
# ------------------------------------------------------------

def retrieve_similar(message, top_k=3):

    message = str(message).strip()

    if not message:
        return []

    query_embedding = model.encode(
        [message],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[int(idx)]

        results.append(
            {
                "customer_message": str(
                    row["customer_message"]
                ),
                "historical_response": str(
                    row["text"]
                ),
                "similarity": float(score)
            }
        )

    return results


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_message = (
        "My order is delayed and has not arrived yet."
    )

    print("=" * 65)
    print("FAST HISTORICAL RETRIEVAL")
    print("=" * 65)

    print("\nCustomer:")
    print(test_message)

    results = retrieve_similar(
        test_message,
        top_k=3
    )

    for result in results:

        print("\nHistorical customer message:")
        print(
            result["customer_message"]
        )

        print("\nHistorical response:")
        print(
            result["historical_response"]
        )

        print("\nSimilarity:")
        print(
            round(
                result["similarity"],
                4
            )
        )

        print("-" * 65)