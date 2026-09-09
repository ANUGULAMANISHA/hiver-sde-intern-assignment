# Hiver SDE Intern Assignment — AI Customer Support Agent

An AI-powered customer support agent built using the **Customer Support on Twitter** dataset for the Hiver SDE Intern take-home assignment.

## Problem

The goal is to build a support agent that can:

1. Classify incoming customer messages into a small set of support intents.
2. Retrieve historically similar customer-support interactions.
3. Draft responses grounded in historical AmazonHelp resolutions.
4. Decide whether a message should be **AUTO-HANDLED** or **ESCALATED** to a human.

## Selected Brand

**AmazonHelp**

The project uses a subset of the Customer Support on Twitter dataset and focuses on AmazonHelp customer-support interactions.

## Support Intents

The current system uses seven intents:

- Account / Contact Support
- Damaged / Defective Product
- Delivery Date / Shipping
- Late / Missing Delivery
- Order Tracking / Status
- Product / Device Support
- Return / Refund

## System Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       +---- TF-IDF + Logistic Regression
       |
       +---- Semantic Embedding Model
       |
       v
Hybrid Intent Prediction
       |
       v
Historical Case Retrieval
       |
       v
Evidence-based Support Decision
       |
       +---- AUTO-HANDLE
       |
       +---- ESCALATE
