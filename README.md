# Hiver SDE Intern Assignment — AI Customer Support Agent

An AI-powered customer support agent built using the **Customer Support on Twitter** dataset for the Hiver SDE Intern take-home assignment.

The system focuses on **AmazonHelp** and implements:

1. Customer-support intent classification
2. Historical-resolution retrieval
3. Evidence-grounded response generation
4. Automatic decision between **AUTO-HANDLE** and **ESCALATE**
5. Evaluation against simple classification baselines
6. A manually labelled golden evaluation set

---

# 1. Problem Framing

The goal is to build a lightweight AI support agent that can understand an incoming customer message, identify its support intent, retrieve relevant historical support interactions, and either:

- generate a safe customer-facing response when sufficient evidence exists, or
- escalate the conversation to a human when confidence, evidence, or risk makes automation unsafe.

## What "good" means

A good support agent should:

- correctly identify the customer's intent
- use historical support interactions as evidence
- avoid inventing policies or actions
- produce concise and useful responses
- avoid claiming access to customer accounts or orders
- escalate uncertain or high-risk cases
- prefer safe escalation over an unsupported answer

## What this project does not attempt

This project is intentionally scoped as a prototype.

It does not attempt to:

- access live Amazon customer accounts
- perform real refunds, cancellations, or order changes
- guarantee production-level intent classification
- learn from private customer information
- replace human support agents
- implement a complete production conversation-management system

---

# 2. Selected Brand

The selected brand is:

**AmazonHelp**

The project uses AmazonHelp interactions from the **Customer Support on Twitter** dataset.

The dataset contains multi-turn customer-support conversations covering many support scenarios.

For this project, the data was reduced to a smaller intent taxonomy suitable for an AI support prototype.

---

# 3. Support Intent Taxonomy

The final system uses seven support intents.

| Intent | Description |
|---|---|
| Account / Contact Support | Account-related problems or requests to contact support |
| Damaged / Defective Product | Damaged, broken, defective, or faulty products |
| Delivery Date / Shipping | Questions about shipping, expected delivery dates, or delivery timing |
| Late / Missing Delivery | Orders that are late, overdue, missing, or have not arrived |
| Order Tracking / Status | Requests to track an order or check current order status |
| Product / Device Support | Questions about products, devices, setup, or usage |
| Return / Refund | Return, refund, replacement, or money-back related requests |

---

# 4. System Architecture

```text
                         Customer Message
                                |
                                v
                    +------------------------+
                    | Intent Classification  |
                    +------------------------+
                         /              \
                        /                \
                       v                  v
             TF-IDF + Logistic     Semantic Embeddings
                Regression              (MiniLM)
                       \                  /
                        \                /
                         v              v
                    +------------------+
                    | Hybrid Classifier|
                    +------------------+
                              |
                              v
                    Historical Retrieval
                              |
                              v
                    Evidence / Confidence
                              |
                    +---------+---------+
                    |                   |
                    v                   v
               AUTO-HANDLE          ESCALATE
                    |                   |
                    v                   v
             Gemini Response       Safe Human-
               Generation          Escalation
```

---

# 5. Data Preparation

The project uses the AmazonHelp portion of the **Customer Support on Twitter** dataset.

The raw conversations were converted into customer-message examples and grouped into a smaller seven-intent taxonomy.

For the classifier experiments, the training pipeline used **17,555 weakly labelled historical examples**.

The weak labels were generated from historical support interactions and message patterns. Because these labels are noisy, the final evaluation was performed separately on a manually labelled golden set.

---

# 6. Intent Classification

Three main approaches were evaluated:

1. Majority-class baseline
2. TF-IDF + Logistic Regression
3. Semantic and hybrid classification

## Baseline 1 — Majority Class

The simplest baseline predicts the most frequent intent for every message.

The majority class was:

**Order Tracking / Status**

Results on the 200-example golden set:

| Metric | Majority Baseline |
|---|---:|
| Accuracy | 0.2200 |
| Macro F1 | 0.0515 |

This provides a useful lower-bound reference because it requires no language model or training.

## Baseline 2 — TF-IDF + Logistic Regression

A conventional text-classification baseline was implemented using:

- TF-IDF features
- unigram and bigram features
- Logistic Regression
- balanced class weights

Results:

| Metric | TF-IDF + Logistic Regression |
|---|---:|
| Accuracy | 0.1450 |
| Macro F1 | 0.1169 |
| Weighted F1 | 0.1005 |

## Semantic Classifier

A SentenceTransformer MiniLM model was also evaluated by comparing message embeddings with historical examples for each intent.

Results:

| Metric | Semantic Classifier |
|---|---:|
| Accuracy | 0.1100 |
| Macro F1 | 0.0800 |
| Weighted F1 | 0.0700 |

## Final Hybrid Classifier

The final classifier combines lexical TF-IDF classification with semantic similarity.

The hybrid score uses:

- **55% TF-IDF classification score**
- **45% semantic similarity score**

The system also applies targeted delivery-related guardrails to distinguish:

- Order Tracking / Status
- Late / Missing Delivery

Results:

| Metric | Hybrid Classifier |
|---|---:|
| Accuracy | 0.1500 |
| Macro F1 | 0.1432 |
| Weighted F1 | 0.1282 |

The hybrid model did not produce the highest accuracy, but it achieved the highest Macro F1 among the learned classifiers tested.

This is important because Macro F1 gives equal importance to performance across the different intents rather than allowing the most frequent intent to dominate the evaluation.

---

# 7. Results Comparison

The main classification comparison is:

| Approach | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority Class | **0.2200** | 0.0515 | — |
| TF-IDF + Logistic Regression | 0.1450 | 0.1169 | 0.1005 |
| Semantic Classifier | 0.1100 | 0.0800 | 0.0700 |
| Hybrid Classifier | 0.1500 | **0.1432** | **0.1282** |

The majority baseline has higher accuracy because the golden set is imbalanced and the majority intent is relatively common.

The learned classifiers perform better in terms of Macro F1, with the hybrid classifier achieving the strongest Macro F1 among the learned approaches.

---

# 8. Golden Evaluation Set

A manually labelled golden set of **200 examples** was created for evaluation.

The examples were sampled from AmazonHelp customer-support messages and assigned one of the seven final intents.

## Golden-set distribution

| Intent | Examples |
|---|---:|
| Account / Contact Support | 16 |
| Damaged / Defective Product | 17 |
| Delivery Date / Shipping | 34 |
| Late / Missing Delivery | 31 |
| Order Tracking / Status | 42 |
| Product / Device Support | 24 |
| Return / Refund | 36 |
| **Total** | **200** |

The golden set was kept separate from the classifier training examples.

The purpose of the golden set is to provide a cleaner estimate of generalisation than the weak labels used during training.

## Sampling and Labelling Note

The golden examples were selected from AmazonHelp customer-support messages to cover the seven target intents.

Each example was manually reviewed and assigned a single intended support intent.

The set intentionally contains difficult boundary cases, particularly around:

- tracking vs late delivery
- delivery date vs delayed delivery
- returns vs refunds
- damaged products vs replacement/refund requests

This makes the evaluation more useful for identifying failure modes than using only easy examples.

---

# 9. Historical Resolution Retrieval

After intent classification, the system retrieves historically similar AmazonHelp interactions.

The retrieval stage is used as evidence for response generation.

The system does not treat a retrieved historical message as an instruction to perform an action.

Instead, retrieved interactions are used to identify previously observed support resolutions and wording.

This reduces the risk of generating unsupported policies.

---

# 10. Response Generation

When the system has sufficient confidence and supporting evidence, it attempts to generate a concise customer-facing response using Gemini.

The response-generation prompt instructs the model to:

- use the retrieved historical evidence
- answer only what the evidence supports
- avoid inventing policies
- avoid claiming access to customer accounts
- avoid claiming that an action was completed
- remain concise and helpful

If generation is unavailable or unsafe, the system falls back to a safe human-support escalation response.

---

# 11. AUTO-HANDLE vs ESCALATE

The agent makes an explicit automation decision after classification and retrieval.

## AUTO-HANDLE

The system can auto-handle when:

- the predicted intent is sufficiently confident
- supporting historical evidence is available
- the request does not require account-specific access
- the response can be grounded in retrieved evidence
- there is no obvious high-risk situation

## ESCALATE

The system escalates when:

- intent confidence is too low
- evidence is insufficient
- the request requires account-specific action
- the system cannot safely generate a grounded answer
- the response-generation service is unavailable

The escalation decision is intentionally conservative.

For example, during an end-to-end test with:

> My order was supposed to arrive yesterday but it still hasn't arrived.

the system classified the message as **Late / Missing Delivery**, retrieved relevant historical evidence, but escalated because the final confidence threshold was not sufficient.

This demonstrates that classification and automation are separate decisions.

---

# 12. Evaluation Harness

The repository contains separate evaluation scripts for:

- baseline classification
- semantic classification
- hybrid classification
- golden-set evaluation
- response evaluation

The classification harness reports:

- accuracy
- Macro F1
- weighted F1
- confusion behaviour
- confidence information

The response evaluation harness defines an LLM-as-judge rubric covering:

1. Correctness
2. Groundedness
3. Helpfulness
4. Safety
5. Conciseness

Each dimension is scored on a 1–5 scale.

An overall response is considered acceptable when it satisfies the defined quality and safety criteria.

The final Gemini judge run was **not used to report headline results** because the available Gemini free-tier quota was exhausted during evaluation. This avoids reporting incomplete or invalid judge scores.

---

# 13. Human Agreement

The response evaluation harness includes fields for human review so that human judgements can be compared with the LLM judge.

The intended evaluation process is:

1. Sample generated responses.
2. Score them using the LLM judge rubric.
3. Independently score the same responses manually.
4. Compare the two judgements.
5. Investigate disagreements.

Because the Gemini quota was exhausted before the final judge run, no unsupported human-vs-LLM agreement number is reported.

This is preferable to presenting an agreement statistic based on an incomplete evaluation.

---

# 14. What Is Misleading About My Headline Number?

The headline classification numbers should **not** be interpreted as production-level accuracy.

There are several reasons:

1. The golden set contains only 200 examples.
2. The training labels are weak/noisy labels derived from historical support interactions.
3. The seven-intent taxonomy is a project-specific simplification.
4. The dataset comes from Twitter support conversations and may not represent all customer-support traffic.
5. Accuracy alone can hide poor performance on individual intents.
6. The hybrid classifier improved Macro F1 relative to the learned baselines, but its overall accuracy was still only 0.15.
7. The golden set was manually labelled for this project rather than being an independently maintained production benchmark.

Therefore, the most useful interpretation is not:

> "The support agent is 15% accurate."

Instead, the result shows that the current prototype has difficulty separating several closely related support intents and needs substantially better labels, taxonomy design, and evaluation data before production use.

---

# 15. Top 5 Failure Modes

## Failure Mode 1 — Delivery Date vs Late Delivery

Messages about delivery timing can look similar even when their intent is different.

Example:

> My order was supposed to arrive yesterday but it still hasn't arrived.

This should be treated as **Late / Missing Delivery**, rather than a general delivery-date question.

### Hypothesis

The historical data contains overlapping language around delivery dates, shipping, tracking, and delays.

---

## Failure Mode 2 — Tracking vs Late/Missing Delivery

A customer asking:

> Where is my order?

is different from:

> My order should have arrived yesterday.

The first is primarily a tracking/status request, while the second indicates a late delivery.

### Hypothesis

Short messages contain too little context for a purely statistical classifier to reliably distinguish these intents.

---

## Failure Mode 3 — Noisy Historical Labels

The training data was created from historical customer-support conversations rather than a clean intent-labelled dataset.

Some examples may therefore contain ambiguous or imperfect labels.

### Hypothesis

The classifier can learn patterns that reflect historical annotation noise rather than the intended seven-intent taxonomy.

---

## Failure Mode 4 — Short or Context-Dependent Messages

Customer-support messages can be extremely short.

Examples such as:

> Help

or:

> Still waiting

do not contain enough information to confidently identify a specific intent.

### Hypothesis

A classifier needs either conversation context or stronger retrieval-based reasoning to handle these messages safely.

---

## Failure Mode 5 — Closely Related Product/Support Requests

Product support, damaged products, returns, refunds, and replacement requests can share vocabulary.

For example, a customer may describe a broken product while also asking for a refund.

### Hypothesis

A single-label taxonomy forces the classifier to choose one dominant intent even when the real customer request contains multiple issues.

---

# 16. Safety Design

The system is deliberately conservative.

It does not claim to:

- access customer accounts
- inspect private order information
- issue refunds
- cancel orders
- change delivery addresses
- complete account actions

When the system cannot safely answer, it escalates.

This is an intentional design choice because an incorrect confident support response can be more harmful than a safe escalation.

---

# 17. Decision Log

The repository contains `decision_log.md` with non-obvious implementation decisions.

Important decisions include:

- selecting AmazonHelp as the single brand
- reducing the dataset to seven intents
- using a manually labelled golden set
- comparing against a majority-class baseline
- using TF-IDF + Logistic Regression as a simple learned baseline
- adding semantic embeddings
- combining lexical and semantic signals
- adding delivery-specific guardrails
- separating classification from automation decisions
- grounding responses in retrieved historical interactions
- using conservative escalation
- avoiding unsupported account/order claims
- not reporting invalid Gemini judge scores after quota exhaustion

---

# 18. Repository Structure

```text
hiver-sde-intern-assignment/
│
├── src/
│   ├── agent.py
│   ├── baseline_classifier.py
│   ├── baselines.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── fast_retrieval.py
│   ├── hybrid_classifier.py
│   ├── hybrid_evaluation.py
│   ├── prepare_training.py
│   ├── reply_evaluation.py
│   ├── retrieval.py
│   └── semantic_evaluation.py
│
├── support_agent.py
├── response_generator.py
├── golden_set.csv
├── golden_evaluation_results.csv
├── semantic_evaluation_results.csv
├── decision_log.md
├── requirements.txt
├── .gitignore
└── README.md
```

Large generated retrieval artifacts such as FAISS indexes are intentionally excluded from Git because of their size.

---

# 19. Reproduction

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure Gemini

Create a local `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or the API key to Git.

## Run the support agent

```bash
py support_agent.py
```

## Run evaluation scripts

```bash
py src/baseline_classifier.py
```

```bash
py src/hybrid_evaluation.py
```

```bash
py src/semantic_evaluation.py
```

```bash
py src/reply_evaluation.py
```

The repository also contains saved evaluation outputs used for the reported classification results.

---

# 20. Limitations

The current prototype has several limitations:

- weakly labelled training data
- relatively small golden evaluation set
- single-brand evaluation
- seven-intent taxonomy
- no live customer account integration
- no production ticketing integration
- no real-time policy database
- limited conversation-context modelling
- limited LLM-as-judge evaluation because of API quota
- retrieval artifacts are generated locally rather than committed to the repository

These limitations are intentional for the scope of the take-home assignment.

---

# 21. What I Would Do With One More Week

With another week, I would prioritise the following.

## 1. Improve the intent taxonomy

Review confusion cases and introduce clearer hierarchical intents, especially around:

- tracking
- late delivery
- shipping dates
- returns
- refunds
- replacements

## 2. Improve labels

Create a larger, independently reviewed evaluation set and improve the quality of training labels.

## 3. Add conversation context

Instead of classifying only one message, include relevant conversation history.

This would help with short messages such as:

> Still waiting.

## 4. Improve retrieval

Use a stronger retrieval/reranking strategy and evaluate retrieval quality independently.

## 5. Improve response evaluation

Run the LLM judge with sufficient quota and complete the human-agreement study.

## 6. Improve reproducibility

Provide a small committed retrieval fixture or a lightweight build script so that a fresh clone can demonstrate the complete pipeline without requiring the full historical index.

---

# 22. Final Summary

This project demonstrates an end-to-end AI customer-support prototype:

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Evidence Retrieval
       |
       v
Confidence + Safety Decision
       |
   +---+---+
   |       |
   v       v
 AUTO    ESCALATE
 HANDLE
   |
   v
Grounded Response
```

The main lesson from the evaluation is that **support intent classification is difficult when historical labels are noisy and intents overlap heavily**.

The prototype therefore prioritises:

- measurable evaluation
- explicit baselines
- evidence-grounded responses
- conservative automation
- transparent escalation
- honest reporting of limitations

---

# 23. References / Attribution

The project uses and builds upon the following public resources and libraries:

- **Customer Support on Twitter dataset**  
  Kaggle dataset: `thoughtvector/customer-support-on-twitter`

- **Sentence-BERT / Sentence Transformers**  
  Reimers, N. and Gurevych, I., *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.

- **scikit-learn**  
  Used for TF-IDF feature extraction and Logistic Regression classification.

- **Sentence Transformers**  
  Used for MiniLM-based semantic sentence embeddings.

- **Google Gemini API**  
  Used for evidence-grounded response generation and the planned LLM-as-judge evaluation.

All implementation code in this repository was developed for this assignment. External libraries are used through their documented APIs.
