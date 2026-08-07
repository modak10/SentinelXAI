# JUDGE_QNA.md

# SentinelXAI Judge Q&A Handbook

Version: 1.0

Competition

ML Bubble 2026

Project

SentinelXAI

Explainable AI Decision Intelligence Platform for Network Intrusion Detection

---

# Purpose

This document prepares the team for technical evaluation.

It contains likely questions judges may ask during

- Project Evaluation
- Technical Interview
- Model Explanation Round
- Live Demonstration

The answers should be understood, not memorized.

---

# Section 1

Project

---

## Q1

What problem does SentinelXAI solve?

Answer

Traditional Intrusion Detection Systems generate thousands of alerts, making it difficult for Security Operations Center analysts to identify genuine threats. Most ML-based IDS projects only predict attack categories without explaining the reasoning behind predictions. SentinelXAI combines machine learning, explainability, confidence estimation, and decision support to help analysts investigate attacks more efficiently.

---

## Q2

Why did you choose cybersecurity?

Answer

Cybersecurity is one of the fastest-growing application areas of machine learning. Network intrusion detection is a real-world problem with publicly available benchmark datasets, making it ideal for developing reproducible and deployable ML solutions.

---

## Q3

What is your project's novelty?

Answer

We are not proposing a new ML algorithm.

Our novelty lies in integrating

- Explainable AI
- Confidence-aware predictions
- Decision Intelligence
- Human-in-the-loop workflow
- Failure analysis
- Production deployment

into a single platform.

---

# Section 2

Dataset

---

## Q4

Why CICIDS2017?

Answer

It is one of the most widely used intrusion detection benchmarks.

Advantages

- Public
- Well documented
- Realistic traffic
- Multiple attack categories
- Rich flow-level features
- Standard benchmark

---

## Q5

Why not use PCAP files?

Answer

PCAP files require feature extraction using tools like CICFlowMeter.

The MachineLearningCSV version already contains extracted flow features, allowing us to focus on machine learning rather than packet parsing.

---

## Q6

How many samples are there?

Answer

Approximately

2.8 million network flows

Around

80 features

Multiple attack classes.

---

## Q7

How did you split the data?

Answer

70%

Training

15%

Validation

15%

Testing

using stratified sampling.

---

## Q8

How did you avoid data leakage?

Answer

Cleaning was performed before splitting only for invalid values.

Feature engineering and preprocessing parameters are learned only from the training data and then applied to validation and test sets.

---

# Section 3

Machine Learning

---

## Q9

Why LightGBM?

Answer

LightGBM performs exceptionally well on structured tabular datasets.

Benefits

Fast

Small model

CPU optimized

High accuracy

Native TreeSHAP support

---

## Q10

Why not Deep Learning?

Answer

Deep learning generally performs better on images, audio, or sequential data.

For structured tabular cybersecurity datasets, gradient boosting methods consistently outperform many neural network architectures while being easier to explain and deploy.

---

## Q11

Why benchmark multiple models?

Answer

Engineering decisions should be evidence-driven.

Benchmarking allows us to justify selecting LightGBM rather than assuming it is the best model.

---

## Q12

Which evaluation metric is most important?

Answer

Macro F1 Score.

Because the dataset is imbalanced, Macro F1 gives equal importance to minority attack classes.

Accuracy alone would be misleading.

---

## Q13

How did you handle class imbalance?

Answer

Class-aware evaluation

Macro F1

Per-class Recall

Optional class weights

Avoid optimizing only accuracy.

---

# Section 4

Explainable AI

---

## Q14

Why SHAP?

Answer

SHAP provides theoretically grounded feature attribution based on cooperative game theory.

TreeSHAP is optimized for tree-based models like LightGBM and produces consistent local explanations.

---

## Q15

What is the difference between local and global explanations?

Answer

Global explanations describe overall model behavior.

Local explanations explain why one specific prediction was made.

---

## Q16

Can SHAP be wrong?

Answer

SHAP explains the model's reasoning, not necessarily reality.

If the model learns incorrect patterns, SHAP will faithfully explain those patterns.

---

# Section 5

Decision Intelligence

---

## Q17

What is the Decision Intelligence Studio?

Answer

It allows users to modify network traffic features interactively and observe changes in prediction, confidence, and SHAP values.

This improves transparency and understanding.

---

## Q18

Why include a Failure Explorer?

Answer

Responsible AI requires transparency.

Instead of hiding incorrect predictions, we expose them to help analysts understand model limitations.

---

## Q19

Why show confidence?

Answer

Predictions with low confidence should receive additional human review.

Confidence helps analysts prioritize investigations.

---

# Section 6

Software Engineering

---

## Q20

Why FastAPI?

Answer

FastAPI is lightweight, fast, asynchronous, and automatically generates OpenAPI documentation.

It is well suited for ML inference APIs.

---

## Q21

Why Streamlit?

Answer

Streamlit enables rapid development of interactive ML dashboards without requiring frontend frameworks.

---

## Q22

Why Docker?

Answer

Docker guarantees reproducibility.

Anyone can run the application regardless of operating system.

---

## Q23

Why SQLite?

Answer

SQLite is lightweight and sufficient for prototype logging and prediction history.

Production systems could migrate to PostgreSQL.

---

# Section 7

Deployment

---

## Q24

Can this be deployed in production?

Answer

Yes.

The modular architecture separates preprocessing, inference, explainability, backend, and frontend, making deployment straightforward.

---

## Q25

How would you scale?

Answer

Future improvements

Load balancing

Redis

PostgreSQL

Kubernetes

Cloud deployment

---

# Section 8

Responsible AI

---

## Q26

How does your project support responsible AI?

Answer

Explainability

Confidence estimation

Failure analysis

Human-in-the-loop workflow

Transparent limitations

---

## Q27

Can this replace analysts?

Answer

No.

SentinelXAI is a decision-support tool.

Final decisions remain with cybersecurity professionals.

---

# Section 9

Future Work

---

## Q28

Future improvements?

Answer

Real-time packet capture

SIEM integration

Model drift detection

Continual learning

Cloud deployment

Threat intelligence feeds

---

## Q29

Would you add LLMs?

Answer

Yes.

LLMs could generate natural-language incident reports, but they would not replace the deterministic ML prediction pipeline.

---

# Section 10

Closing Questions

---

## Q30

If you had another month?

Answer

Cross-dataset validation

MLflow

Drift monitoring

Role-based authentication

Cloud deployment

Real-time streaming

---

## Q31

What did you learn?

Answer

We learned that building production-quality ML systems involves much more than training models.

Understanding data engineering, explainability, deployment, and human-centered design is equally important.

---

# Questions You Should Ask Judges

At the end of the session

"What deployment considerations would you recommend for integrating such a system into enterprise SOC environments?"

This demonstrates maturity and genuine interest.

---

# Final Advice

Remember

We are not defending a classifier.

We are defending a complete Machine Learning product.

Every answer should reinforce four key themes

- Explainability
- Engineering Quality
- Human-centered AI
- Production Readiness

If judges remember only one sentence, let it be

"SentinelXAI is not another intrusion detection model; it is an Explainable AI Decision Intelligence Platform that helps analysts make faster, more transparent, and more trustworthy cybersecurity decisions."