import pandas as pd
import re

INPUT_FILE = "amazonhelp_pairs.csv"
OUTPUT_FILE = "training_data.csv"

print("=" * 60)
print("PREPARING HISTORICAL TRAINING DATA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Historical rows:", len(df))
print("Columns:", df.columns.tolist())


def classify_intent(text):
    text = str(text).lower()

    # Account / contact support
    if any(x in text for x in [
        "account", "password", "login", "sign in",
        "email", "e-mail", "contact", "security"
    ]):
        return "Account / Contact Support"

    # Damaged / defective
    if any(x in text for x in [
        "damaged", "broken", "defective", "faulty",
        "not working", "doesn't work", "doesnt work"
    ]):
        return "Damaged / Defective Product"

    # Return / refund
    if any(x in text for x in [
        "refund", "return", "money back", "cancel my order",
        "cancel order"
    ]):
        return "Return / Refund"

    # Late / missing delivery
    if any(x in text for x in [
        "late", "delayed", "delay", "not arrived",
        "haven't received", "have not received",
        "missing", "where is my order"
    ]):
        return "Late / Missing Delivery"

    # Tracking / status
    if any(x in text for x in [
        "tracking", "track my", "track order",
        "order status", "status of my order",
        "where is my package", "where is my parcel"
    ]):
        return "Order Tracking / Status"

    # Delivery / shipping
    if any(x in text for x in [
        "delivery date", "deliver", "shipping",
        "ship", "shipping date", "when will"
    ]):
        return "Delivery Date / Shipping"

    # Product / device support
    if any(x in text for x in [
        "how do i use", "how to use", "setup",
        "set up", "compatible", "device", "echo",
        "kindle", "fire", "alexa", "app"
    ]):
        return "Product / Device Support"

    return None


# Use customer messages only.
texts = df["customer_message"].fillna("").astype(str)

df["intent"] = texts.apply(classify_intent)

# Remove examples that cannot be confidently assigned.
training = df[
    df["intent"].notna() &
    (texts.str.strip() != "")
].copy()

training = training[
    ["customer_message", "intent"]
].rename(
    columns={"customer_message": "text"}
)

# Limit each class so one intent does not dominate.
MAX_PER_INTENT = 3000

training = (
    training
    .groupby("intent", group_keys=False)
    .apply(
        lambda x: x.sample(
            min(len(x), MAX_PER_INTENT),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

training.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTraining data created:", len(training))

print("\nIntent distribution:")
print(training["intent"].value_counts())

print("\nSaved:")
print(OUTPUT_FILE)