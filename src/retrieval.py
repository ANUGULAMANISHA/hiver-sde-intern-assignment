import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Load historical AmazonHelp conversations
df = pd.read_csv("amazonhelp_pairs.csv")

df = df.dropna(subset=["customer_message", "text"]).copy()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for historical customer messages
print("Creating embeddings...")

embeddings = model.encode(
    df["customer_message"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Embeddings created:", len(embeddings))


def retrieve_similar(message, top_k=5):
    """Find historically similar customer messages."""

    query_embedding = model.encode(
        message,
        convert_to_tensor=True
    )

    scores = util.cos_sim(query_embedding, embeddings)[0]

    top_results = scores.topk(k=min(top_k, len(df)))

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        row = df.iloc[int(index)]

        results.append({
            "customer_message": row["customer_message"],
            "historical_response": row["text"],
            "similarity": float(score)
        })

    return results


# Test the retriever
test_message = (
    "My order has not arrived yet and the delivery is delayed."
)

print("\n=== Similar Historical Cases ===")

for result in retrieve_similar(test_message):

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nAmazonHelp response:")
    print(result["historical_response"])

    print("\nSimilarity:", round(result["similarity"], 4))