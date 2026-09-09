import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# Load saved data and FAISS index
df = pd.read_csv("amazonhelp_retrieval_data.csv")
index = faiss.read_index("amazonhelp.index")

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_similar(message, top_k=5):
    query_embedding = model.encode(
        [message],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[int(idx)]

        results.append({
            "customer_message": row["customer_message"],
            "historical_response": row["text"],
            "similarity": float(score)
        })

    return results


# Test
message = "My order is delayed and has not arrived yet."

print("=== FAST HISTORICAL RETRIEVAL ===")

for result in retrieve_similar(message):

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nHistorical response:")
    print(result["historical_response"])

    print("\nSimilarity:", round(result["similarity"], 4))