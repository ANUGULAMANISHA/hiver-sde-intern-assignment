from src.hybrid_classifier import hybrid_predict
from src.fast_retrieval import retrieve_similar
from response_generator import generate_reply


# ============================================================
# Hiver SDE Assignment - End-to-End AI Support Agent
# ============================================================

INTENT_CONFIDENCE_THRESHOLD = 0.60
HISTORICAL_EVIDENCE_THRESHOLD = 0.60


# Intents requiring human verification
RISKY_INTENTS = {
    "Account / Contact Support",
    "Return / Refund"
}


# ------------------------------------------------------------
# Decision logic
# ------------------------------------------------------------

def make_decision(
    intent,
    intent_confidence,
    historical_cases
):
    """
    Decide whether a request should be AUTO-HANDLED
    or ESCALATED to a human.
    """

    reasons = []


    # Low intent confidence
    if intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
        reasons.append(
            "low intent confidence"
        )


    # No historical evidence
    if not historical_cases:
        reasons.append(
            "no historical evidence found"
        )

    else:

        best_similarity = historical_cases[0]["similarity"]

        if best_similarity < HISTORICAL_EVIDENCE_THRESHOLD:
            reasons.append(
                "weak historical evidence"
            )


    # Risky intents
    if intent in RISKY_INTENTS:
        reasons.append(
            "requires human verification"
        )


    if reasons:

        return {
            "decision": "ESCALATE",
            "reason": "; ".join(reasons)
        }


    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            "high-confidence intent with "
            "relevant historical evidence"
        )
    }


# ------------------------------------------------------------
# Complete support agent
# ------------------------------------------------------------

def support_agent(message):
    """
    Complete Hiver support-agent pipeline:

    1. Intent classification
    2. Historical retrieval
    3. Escalation decision
    4. Gemini response generation only for AUTO-HANDLE
    """

    message = str(message).strip()


    if not message:

        return {
            "customer_message": "",
            "intent": "Unknown",
            "confidence": 0.0,
            "tfidf_intent": "Unknown",
            "tfidf_confidence": 0.0,
            "semantic_intent": "Unknown",
            "semantic_confidence": 0.0,
            "historical_cases": [],
            "decision": "ESCALATE",
            "reason": "empty customer message",
            "reply": (
                "Please provide more details about "
                "your issue so we can assist you."
            )
        }


    # ========================================================
    # 1. Hybrid classification
    # ========================================================

    classification = hybrid_predict(
        message
    )


    intent = classification["intent"]

    confidence = classification["confidence"]


    # ========================================================
    # 2. Historical retrieval
    # ========================================================

    historical_cases = retrieve_similar(
        message,
        top_k=3
    )


    # ========================================================
    # 3. Decision
    # ========================================================

    decision_result = make_decision(
        intent=intent,
        intent_confidence=confidence,
        historical_cases=historical_cases
    )


    decision = decision_result["decision"]

    reason = decision_result["reason"]


    # ========================================================
    # 4. Response generation
    # ========================================================

    if decision == "ESCALATE":

        reply = (
            "This request requires human customer-support "
            "assistance. Please contact AmazonHelp so that "
            "a support representative can help you."
        )

    else:

        try:

            reply = generate_reply(
                customer_message=message,
                historical_cases=historical_cases,
                intent=intent
            )

        except Exception as error:

            # If Gemini is temporarily unavailable,
            # safely escalate instead of failing.
            decision = "ESCALATE"

            reason = (
                "response generation unavailable; "
                "human support recommended"
            )

            reply = (
                "I'm sorry, but I'm unable to provide a "
                "reliable response right now. Please contact "
                "customer support for assistance."
            )


    # ========================================================
    # 5. Final result
    # ========================================================

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": confidence,
        "tfidf_intent": classification["tfidf_intent"],
        "tfidf_confidence": classification["tfidf_confidence"],
        "semantic_intent": classification["semantic_intent"],
        "semantic_confidence": classification["semantic_confidence"],
        "historical_cases": historical_cases,
        "decision": decision,
        "reason": reason,
        "reply": reply
    }


# ============================================================
# Display result
# ============================================================

def display_result(result):

    print("\n")
    print("=" * 70)
    print("HIVER AI CUSTOMER SUPPORT AGENT")
    print("=" * 70)


    print("\nCustomer message:")
    print(
        result["customer_message"]
    )


    print("\nPredicted intent:")
    print(
        result["intent"]
    )


    print(
        f"Intent confidence: "
        f"{result['confidence']:.4f}"
    )


    print("\nTF-IDF prediction:")
    print(
        f"{result['tfidf_intent']} "
        f"({result['tfidf_confidence']:.4f})"
    )


    print("\nSemantic prediction:")
    print(
        f"{result['semantic_intent']} "
        f"({result['semantic_confidence']:.4f})"
    )


    print("\nHistorical evidence:")

    if not result["historical_cases"]:

        print(
            "No historical cases found."
        )

    else:

        for i, case in enumerate(
            result["historical_cases"],
            start=1
        ):

            print(
                f"\nCase {i}"
            )

            print("Customer:")
            print(
                case["customer_message"]
            )

            print("Historical response:")
            print(
                case["historical_response"]
            )

            print(
                "Similarity:",
                round(
                    case["similarity"],
                    4
                )
            )


    print("\nDecision:")
    print(
        result["decision"]
    )


    print("\nReason:")
    print(
        result["reason"]
    )


    print("\nAI Support Reply:")
    print(
        result["reply"]
    )


    print("\n" + "=" * 70)


# ============================================================
# Interactive mode
# ============================================================

def interactive_mode():

    print("\n" + "=" * 70)
    print("HIVER AI SUPPORT AGENT - INTERACTIVE MODE")
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


        try:

            result = support_agent(
                message
            )

            display_result(
                result
            )

        except Exception as error:

            print(
                "\nError while processing "
                "customer message:"
            )

            print(error)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_message = (
        "My order is delayed and has not arrived yet. "
        "Can you tell me what is happening?"
    )


    print(
        "\nRunning end-to-end test..."
    )


    result = support_agent(
        test_message
    )


    display_result(
        result
    )


    interactive_mode()