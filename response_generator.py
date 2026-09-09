import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.7-flash"


def create_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your local .env file."
        )

    return genai.Client(api_key=api_key)


def generate_reply(customer_message, historical_cases):
    client = create_client()

    evidence = "\n\n".join(
        [
            f"Historical customer message: {case['customer_message']}\n"
            f"Historical response: {case['historical_response']}"
            for case in historical_cases
        ]
    )

    prompt = f"""
You are an AI customer-support assistant for AmazonHelp.

Customer message:
{customer_message}

Historical support examples:
{evidence}

Write a short, helpful customer-facing reply.

Rules:
- Ground the reply in the historical support examples.
- Do not invent refund policies, compensation, delivery dates, or account actions.
- Do not claim that you accessed the customer's account or order.
- If the historical evidence is insufficient, recommend human support instead of guessing.
- Do not mention similarity scores, classification, retrieval, or internal systems.
- Do not expose private information.
- Be professional, natural, and concise.
- Return ONLY the customer-facing reply.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


if __name__ == "__main__":
    test_message = (
        "My order is delayed and has not arrived yet. "
        "Can you tell me what is happening?"
    )

    historical_cases = [
        {
            "customer_message": "My package is late and has not arrived.",
            "historical_response": (
                "We're sorry your package hasn't arrived yet. "
                "Please check your tracking information for the latest update."
            ),
        },
        {
            "customer_message": "My delivery is delayed.",
            "historical_response": (
                "We apologize for the delay. Please use the tracking "
                "information associated with your order to check its status."
            ),
        },
    ]

    reply = generate_reply(test_message, historical_cases)

    print("\nCustomer:")
    print(test_message)

    print("\nAI Support Reply:")
    print(reply)