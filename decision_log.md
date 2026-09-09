# Decision Log

## 1. Selected AmazonHelp as the target brand
I selected AmazonHelp because the dataset contains a substantial number of Amazon customer-support interactions, providing enough historical examples for classification and retrieval.

## 2. Defined a small intent taxonomy
I used seven practical support intents:
- Account / Contact Support
- Damaged / Defective Product
- Delivery Date / Shipping
- Late / Missing Delivery
- Order Tracking / Status
- Product / Device Support
- Return / Refund

A compact taxonomy makes classification more reliable and easier to evaluate.

## 3. Created a manually labelled golden set
I created a 200-example golden evaluation set and labelled each example using the defined intent taxonomy.

## 4. Kept the golden set separate from training data
The golden set is evaluated separately from the historical training data to reduce evaluation leakage.

## 5. Used TF-IDF + Logistic Regression as a simple baseline
This provides a transparent and reproducible text-classification baseline against which more semantic approaches can be compared.

## 6. Added semantic embeddings
I used Sentence Transformers to capture semantic similarity that keyword-based TF-IDF may miss.

## 7. Added FAISS for historical retrieval
FAISS provides efficient similarity search over historical customer-support interactions.

## 8. Combined lexical and semantic signals
The hybrid classifier considers both TF-IDF classification and semantic predictions because either signal alone can fail on ambiguous customer messages.

## 9. Used historical responses as evidence
Retrieved historical AmazonHelp responses are used as evidence for support decisions rather than generating unsupported solutions.

## 10. Added confidence-based escalation
Low-confidence predictions are escalated instead of being automatically handled.

## 11. Added historical-evidence thresholds
Even when intent confidence is high, weak historical similarity can trigger escalation because the system may lack sufficient evidence for a reliable response.

## 12. Escalated account and refund requests
Account/contact and return/refund requests can require customer-specific verification, so the system conservatively routes them to human support.

## 13. Evaluated on an unseen golden set
The final evaluation uses examples not used to train the classifier, providing a more realistic estimate of generalization.

## 14. Reported both accuracy and Macro-F1
Accuracy alone can hide poor performance on individual intents. Macro-F1 gives each intent equal importance and is therefore included as a primary evaluation metric.

## 15. Kept the full Twitter dataset outside Git
The original dataset and large generated indexes are excluded from the repository because they are large and are not required for reviewers to inspect the implementation.
