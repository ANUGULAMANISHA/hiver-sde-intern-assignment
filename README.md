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

## 1. Problem Framing

The goal is to build a lightweight AI support agent that can understand an incoming customer message, identify its support intent, retrieve relevant historical support interactions, and either:

- generate a safe customer-facing response when sufficient evidence exists, or
- escalate the conversation to a human when confidence, evidence, or risk makes automation unsafe.

### What "good" means

A good support agent should:

- correctly identify the customer's intent
- use historical support interactions as evidence
- avoid inventing policies or actions
- produce concise and useful responses
- avoid claiming access to customer accounts or orders
- escalate uncertain or high-risk cases
- prefer safe escalation over an unsupported answer

### What this project does not attempt

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
                    |   Intent Classification |
                    +------------------------+
                         /              \
                        /                \
                       v                  v
             TF-IDF + Logistic     Semantic Embeddings
                Regression          (MiniLM)
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
