import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import cohen_kappa_score

from support_agent import support_agent


# ============================================================
# Hiver SDE Assignment
# Reply Quality Evaluation + LLM-as-Judge
# ============================================================

load_dotenv()

MODEL_NAME = "gemini-3.7-flash"

GOLDEN_FILE = "golden_set.csv"
RESULT_FILE = "reply_judge_results.csv"

# One example from each of the 7 intents
# This keeps the evaluation small enough for the free API quota.
SAMPLE_PER_INTENT = 1


# ------------------------------------------------------------
# Gemini client
# ------------------------------------------------------------

def create_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set in .env"
        )

    return genai.Client(
        api_key=api_key
    )


# ------------------------------------------------------------
# Load golden set
# ------------------------------------------------------------

def load_golden_set():

    df = pd.read_csv(
        GOLDEN_FILE
    )

    required_columns = [
        "customer_message",
        "intent"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

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

    return df


# ------------------------------------------------------------
# Stratified evaluation sample
# ------------------------------------------------------------

def create_evaluation_sample(df):

    samples = []

    for intent in sorted(
        df["intent"].unique()
    ):

        intent_rows = df[
            df["intent"] == intent
        ]

        selected = intent_rows.sample(
            n=min(
                SAMPLE_PER_INTENT,
                len(intent_rows)
            ),
            random_state=42
        )

        samples.append(
            selected
        )

    if not samples:
        raise ValueError(
            "No evaluation examples found."
        )

    sample = pd.concat(
        samples,
        ignore_index=True
    )

    return sample


# ------------------------------------------------------------
# Automated reply metrics
# ------------------------------------------------------------

def calculate_automated_metrics(
    customer_message,
    reply,
    historical_cases
):
    """
    Lightweight automated metrics that do not require
    another model.

    These are diagnostic metrics, not human-quality labels.
    """

    reply = str(reply).strip()

    customer_words = set(
        customer_message.lower().split()
    )

    reply_words = set(
        reply.lower().split()
    )

    # Customer/reply lexical overlap
    if customer_words:

        overlap = (
            len(
                customer_words
                & reply_words
            )
            / len(customer_words)
        )

    else:

        overlap = 0.0

    # Historical evidence strength
    if historical_cases:

        best_similarity = max(
            case.get(
                "similarity",
                0.0
            )
            for case in historical_cases
        )

    else:

        best_similarity = 0.0

    # Basic response safety checks
    forbidden_terms = [
        "similarity score",
        "embedding",
        "faiss",
        "classification",
        "internal system"
    ]

    forbidden_found = any(
        term in reply.lower()
        for term in forbidden_terms
    )

    # Basic length metric
    word_count = len(
        reply.split()
    )

    return {
        "reply_word_count": word_count,

        "customer_reply_overlap": round(
            overlap,
            4
        ),

        "best_historical_similarity": round(
            best_similarity,
            4
        ),

        "forbidden_internal_terms": int(
            forbidden_found
        )
    }


# ------------------------------------------------------------
# LLM judge
# ------------------------------------------------------------

def judge_reply(
    client,
    customer_message,
    predicted_intent,
    decision,
    reply,
    historical_cases
):
    """
    Ask Gemini to judge reply quality.

    Rubric:
    1. Correctness
    2. Groundedness
    3. Helpfulness
    4. Safety
    5. Conciseness

    Each dimension is scored 1-5.
    """

    evidence_text = ""

    for i, case in enumerate(
        historical_cases[:3],
        start=1
    ):

        evidence_text += f"""
Historical case {i}

Customer:
{case["customer_message"]}

Historical response:
{case["historical_response"]}
"""

    if not evidence_text:

        evidence_text = (
            "No historical evidence was retrieved."
        )

    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

System decision:
{decision}

AI-generated reply:
{reply}

Historical evidence used by the system:
{evidence_text}

Evaluate the reply using this rubric.

CORRECTNESS:
Does the response appropriately address the customer's
actual issue without making unsupported claims?

GROUNDedness:
Is the response consistent with the historical support
evidence provided?

HELPFULNESS:
Would this response be useful to the customer?

SAFETY:
Does it avoid inventing refunds, policies, account actions,
delivery promises, compensation, or access to private data?

CONCISENESS:
Is it reasonably short and direct for customer support?

Score each dimension from 1 to 5:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Then calculate the overall score as the arithmetic mean.

Also classify the reply as:

ACCEPTABLE
if the overall quality is 4 or 5.

UNACCEPTABLE
if the overall quality is 1, 2, or 3.

Return ONLY valid JSON in exactly this structure:

{{
  "correctness": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "safety": 1,
  "conciseness": 1,
  "overall": 1.0,
  "label": "ACCEPTABLE",
  "reason": "short explanation"
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = response.text.strip()

    # Remove accidental markdown fences
    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    return json.loads(
        text
    )


# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------

def run_evaluation():

    print("=" * 70)
    print("HIVER REPLY QUALITY EVALUATION")
    print("=" * 70)

    golden = load_golden_set()

    print(
        f"\nGolden examples available: "
        f"{len(golden)}"
    )

    sample = create_evaluation_sample(
        golden
    )

    print(
        f"Evaluation sample size: "
        f"{len(sample)}"
    )

    print("\nSample distribution:")

    print(
        sample["intent"].value_counts()
    )

    client = create_client()

    results = []

    for position, row in sample.iterrows():

        customer_message = (
            row["customer_message"]
        )

        expected_intent = (
            row["intent"]
        )

        print(
            f"\n[{position + 1}/{len(sample)}]"
        )

        print(
            "Customer:",
            customer_message
        )

        try:

            # ------------------------------------------------
            # Run the actual support agent
            # ------------------------------------------------

            agent_result = support_agent(
                customer_message
            )

            reply = agent_result["reply"]

            predicted_intent = (
                agent_result["intent"]
            )

            decision = (
                agent_result["decision"]
            )

            historical_cases = (
                agent_result["historical_cases"]
            )

            # ------------------------------------------------
            # Automated diagnostics
            # ------------------------------------------------

            automated = calculate_automated_metrics(
                customer_message,
                reply,
                historical_cases
            )

            # ------------------------------------------------
            # LLM judge
            # ------------------------------------------------

            judge = judge_reply(
                client=client,
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                decision=decision,
                reply=reply,
                historical_cases=historical_cases
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            result = {

                "customer_message":
                    customer_message,

                "gold_intent":
                    expected_intent,

                "predicted_intent":
                    predicted_intent,

                "intent_correct":
                    int(
                        expected_intent
                        == predicted_intent
                    ),

                "decision":
                    decision,

                "reason":
                    agent_result["reason"],

                "reply":
                    reply,

                "correctness":
                    judge["correctness"],

                "groundedness":
                    judge["groundedness"],

                "helpfulness":
                    judge["helpfulness"],

                "safety":
                    judge["safety"],

                "conciseness":
                    judge["conciseness"],

                "judge_overall":
                    judge["overall"],

                "judge_label":
                    judge["label"],

                "judge_reason":
                    judge["reason"],

                "human_overall":
                    "",

                "human_label":
                    "",

                "human_reason":
                    "",

                "reply_word_count":
                    automated[
                        "reply_word_count"
                    ],

                "customer_reply_overlap":
                    automated[
                        "customer_reply_overlap"
                    ],

                "best_historical_similarity":
                    automated[
                        "best_historical_similarity"
                    ],

                "forbidden_internal_terms":
                    automated[
                        "forbidden_internal_terms"
                    ]
            }

            results.append(
                result
            )

            print(
                "Predicted intent:",
                predicted_intent
            )

            print(
                "Decision:",
                decision
            )

            print(
                "Judge label:",
                judge["label"]
            )

            print(
                "Judge overall:",
                judge["overall"]
            )

        except Exception as error:

            print(
                "Evaluation error:",
                error
            )

            results.append(
                {
                    "customer_message":
                        customer_message,

                    "gold_intent":
                        expected_intent,

                    "predicted_intent":
                        "",

                    "intent_correct":
                        "",

                    "decision":
                        "",

                    "reason":
                        "",

                    "reply":
                        "",

                    "correctness":
                        "",

                    "groundedness":
                        "",

                    "helpfulness":
                        "",

                    "safety":
                        "",

                    "conciseness":
                        "",

                    "judge_overall":
                        "",

                    "judge_label":
                        "",

                    "judge_reason":
                        str(error),

                    "human_overall":
                        "",

                    "human_label":
                        "",

                    "human_reason":
                        "",

                    "reply_word_count":
                        "",

                    "customer_reply_overlap":
                        "",

                    "best_historical_similarity":
                        "",

                    "forbidden_internal_terms":
                        ""
                }
            )

        # ----------------------------------------------------
        # API rate-limit protection
        # ----------------------------------------------------

        print(
            "Waiting 20 seconds before the next example..."
        )

        time.sleep(20)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved results to: "
        f"{RESULT_FILE}"
    )

    # --------------------------------------------------------
    # Calculate successful evaluations
    # --------------------------------------------------------

    successful = results_df[
        results_df["judge_overall"]
        .astype(str)
        .str.strip()
        != ""
    ]

    if len(successful) > 0:

        print(
            "\nAverage judge scores:"
        )

        for column in [
            "correctness",
            "groundedness",
            "helpfulness",
            "safety",
            "conciseness",
            "judge_overall"
        ]:

            values = pd.to_numeric(
                successful[column],
                errors="coerce"
            )

            print(
                f"{column}: "
                f"{values.mean():.2f}"
            )

        judge_acceptable = (
            successful["judge_label"]
            == "ACCEPTABLE"
        ).mean()

        print(
            f"\nJudge acceptable rate: "
            f"{judge_acceptable:.2%}"
        )

    else:

        print(
            "\nNo successful LLM-judge evaluations."
        )


# ------------------------------------------------------------
# Human agreement calculation
# ------------------------------------------------------------

def calculate_human_agreement():

    if not os.path.exists(
        RESULT_FILE
    ):

        print(
            "Result file not found."
        )

        print(
            "Run the evaluation first."
        )

        return

    df = pd.read_csv(
        RESULT_FILE
    )

    print("=" * 70)
    print("HUMAN VS LLM JUDGE AGREEMENT")
    print("=" * 70)

    print(
        "\nHuman labels required."
    )

    print(
        "Open reply_judge_results.csv "
        "and fill the human_label column."
    )

    print(
        "Use only:"
    )

    print(
        "ACCEPTABLE"
    )

    print(
        "or"
    )

    print(
        "UNACCEPTABLE"
    )

    valid = df[
        df["judge_label"].isin(
            [
                "ACCEPTABLE",
                "UNACCEPTABLE"
            ]
        )
        &
        df["human_label"].isin(
            [
                "ACCEPTABLE",
                "UNACCEPTABLE"
            ]
        )
    ].copy()

    if len(valid) < 2:

        print(
            "\nNot enough human labels yet."
        )

        print(
            "Fill human_label for at least "
            "2 examples, then run again."
        )

        return

    judge_labels = (
        valid["judge_label"]
        .astype(str)
    )

    human_labels = (
        valid["human_label"]
        .astype(str)
    )

    agreement = (
        judge_labels
        == human_labels
    ).mean()

    kappa = cohen_kappa_score(
        human_labels,
        judge_labels
    )

    print(
        f"\nExamples with both labels: "
        f"{len(valid)}"
    )

    print(
        f"Raw agreement: "
        f"{agreement:.2%}"
    )

    print(
        f"Cohen's kappa: "
        f"{kappa:.4f}"
    )

    print(
        "\nThis is the evidence to report for "
        "LLM-judge vs human agreement."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "\nChoose an option:"
    )

    print(
        "1 - Run reply evaluation"
    )

    print(
        "2 - Calculate human vs LLM agreement"
    )

    choice = input(
        "\nEnter 1 or 2: "
    ).strip()

    if choice == "1":

        run_evaluation()

    elif choice == "2":

        calculate_human_agreement()

    else:

        print(
            "Invalid option."
        )