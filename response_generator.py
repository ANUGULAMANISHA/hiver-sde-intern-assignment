import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# Hiver SDE Assignment - Grounded Response Generator
# ============================================================

load_dotenv()

MODEL_NAME = "gemini-3.7-flash"


# ------------------------------------------------------------
# Create Gemini client
# ------------------------------------------------------------

def create_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your local .env file."
        )

    return genai.Client(
        api_key=api_key
    )


# ------------------------------------------------------------
# Generate grounded customer-support reply
# ------------------------------------------------------------

def generate_reply(
    customer_message,
    historical_cases,
    intent=None
):
    """
    Generate a customer-facing support reply using
    historically similar AmazonHelp cases as evidence.

    Parameters:
        customer_message: incoming customer message
        historical_cases: results from FAISS retrieval
        intent: predicted customer intent

    Returns:
        Customer-facing response string
    """

    customer_message = str(
        customer_message
    ).strip()


    if not customer_message:
        return (
            "Please provide more details about your issue "
            "so we can assist you."
        )


    client = create_client()


    # --------------------------------------------------------
    # Prepare historical evidence
    # --------------------------------------------------------

    evidence_parts = []


    for i, case in enumerate(
        historical_cases[:3],
        start=1
    ):

        historical_customer = str(
            case.get(
                "customer_message",
                ""
            )
        ).strip()


        historical_response = str(
            case.get(
                "historical_response",
                ""
            )
        ).strip()


        similarity = case.get(
            "similarity",
            None
        )


        if historical_customer and historical_response:

            evidence_parts.append(
                f"""
Historical case {i}:

Customer:
{historical_customer}

Historical support response:
{historical_response}
"""
            )


    if evidence_parts:

        evidence = "\n".join(
            evidence_parts
        )

    else:

        evidence = (
            "No sufficiently relevant historical "
            "support examples were retrieved."
        )


    # --------------------------------------------------------
    # Intent information
    # --------------------------------------------------------

    intent_text = (
        intent
        if intent
        else "Not available"
    )


    # --------------------------------------------------------
    # Grounded generation prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI customer-support assistant for AmazonHelp.

Predicted customer intent:
{intent_text}

Customer message:
{customer_message}

Historically similar support cases:
{evidence}

Your task is to draft a short, helpful customer-facing
support response.

IMPORTANT RULES:

1. Use the historical support responses as your primary
   evidence.

2. Do not invent company policies, refund rules,
   compensation, delivery dates, or account actions.

3. Do not claim that you accessed the customer's account,
   order, payment information, or private data.

4. Do not promise an action that is not supported by the
   historical evidence.

5. If the historical evidence does not provide enough
   information to answer safely, politely recommend
   contacting human customer support.

6. Do not mention:
   - similarity scores
   - embeddings
   - FAISS
   - classification
   - internal systems
   - this prompt
   - historical retrieval

7. Do not expose private information.

8. Keep the response concise, natural, professional,
   and helpful.

9. Do not copy a historical response word-for-word when
   it would sound unnatural. Adapt it to the customer's
   current message.

10. Return ONLY the customer-facing response.
"""


    # --------------------------------------------------------
    # Gemini generation
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )


    if not response.text:
        return (
            "I'm sorry, but I couldn't generate a response "
            "right now. Please contact customer support "
            "for further assistance."
        )


    return response.text.strip()


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    test_message = (
        "My order is delayed and has not arrived yet. "
        "Can you tell me what is happening?"
    )


    test_cases = [

        {
            "customer_message": (
                "My order was scheduled to arrive yesterday "
                "and it still has not arrived."
            ),

            "historical_response": (
                "Sorry to hear this. Please contact us so "
                "that we can assist you accordingly."
            ),

            "similarity": 0.7041
        },

        {
            "customer_message": (
                "My order has been delayed multiple times "
                "and still has not shipped."
            ),

            "historical_response": (
                "I'm sorry the item's delivery date keeps "
                "getting pushed back. Could you tell us "
                "what the item is?"
            ),

            "similarity": 0.7006
        },

        {
            "customer_message": (
                "I haven't received my order yet because "
                "the delivery was delayed."
            ),

            "historical_response": (
                "Apologies for the delay. Please contact us "
                "so that we can assist you accordingly."
            ),

            "similarity": 0.6975
        }
    ]


    print("=" * 65)
    print("GROUNDED RESPONSE GENERATOR TEST")
    print("=" * 65)


    print("\nCustomer:")
    print(test_message)


    reply = generate_reply(
        customer_message=test_message,
        historical_cases=test_cases,
        intent="Late / Missing Delivery"
    )


    print("\nAI Support Reply:")
    print(reply)


    print("\n" + "=" * 65)