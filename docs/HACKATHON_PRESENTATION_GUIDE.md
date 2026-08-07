# HACKATHON_PRESENTATION_GUIDE.md

# SentinelXAI Presentation Guide

Version: 1.0

Competition

ML Bubble 2026

Presentation Time

7–10 Minutes

Project

SentinelXAI
Explainable AI Decision Intelligence Platform

---

# Objective

The goal of this presentation is NOT to convince judges that we trained a good model.

The goal is to convince judges that

- We understand Machine Learning
- We understand Software Engineering
- We understand Explainable AI
- We understand Deployment
- We understand Product Design
- We understand Real-world Problems

Accuracy is only one part of the evaluation.

---

# Presentation Strategy

Our presentation follows a story.

Problem

↓

Why existing solutions fail

↓

Our solution

↓

Machine Learning

↓

Explainability

↓

Deployment

↓

Impact

↓

Future

---

# Time Allocation

| Section | Time |
|----------|------|
| Introduction | 45 sec |
| Problem | 1 min |
| Solution | 1 min |
| ML Pipeline | 1 min |
| Live Demo | 3 min |
| Architecture | 45 sec |
| Results | 45 sec |
| Future Scope | 30 sec |
| Questions | Remaining |

---

# Slide 1

Title

SentinelXAI

Subtitle

Explainable AI Decision Intelligence Platform for Network Intrusion Detection

Say

"Good morning everyone.

Cybersecurity teams receive thousands of security alerts every day.

Our project helps analysts understand which attacks matter most, why the AI predicted them, and what action they should take."

---

# Slide 2

Problem

Illustration

SOC Analyst

↓

Thousands of Alerts

↓

Alert Fatigue

↓

False Positives

↓

Slow Investigation

Say

"Most ML IDS projects stop after predicting an attack.

Real analysts need much more than a label."

---

# Slide 3

Current Limitations

Traditional ML

↓

Prediction

↓

Done

SentinelXAI

↓

Prediction

↓

Confidence

↓

Explainability

↓

Recommendation

↓

Decision Support

Key Message

"We are building a decision-support platform, not another classifier."

---

# Slide 4

Dataset

CICIDS2017

Explain

- Public benchmark
- Enterprise traffic
- Multiple attack classes
- Standard research dataset

Mention

MachineLearningCSV

Not PCAP

---

# Slide 5

Machine Learning Pipeline

Diagram

Dataset

↓

Cleaning

↓

Feature Engineering

↓

LightGBM

↓

SHAP

↓

Decision Intelligence

↓

Dashboard

Important

Mention baseline models first.

Explain why LightGBM was selected.

---

# Slide 6

Architecture

Show

Streamlit

↓

FastAPI

↓

LightGBM

↓

SHAP

↓

SQLite

Judges love architecture diagrams.

Spend no more than

45 seconds.

---

# Slide 7

Live Demo ⭐⭐⭐⭐⭐

This decides the competition.

Workflow

Upload CSV

↓

Predict

↓

Explain

↓

Decision Studio

↓

Failure Explorer

Practice this repeatedly.

---

# Demo Script

Step 1

Upload sample CSV

Say

"Let's analyze real network traffic."

---

Step 2

Prediction

Show

Botnet

98%

Critical

Say

"The model predicts Botnet with very high confidence."

---

Step 3

Explainability

Open SHAP

Say

"Instead of hiding the decision, we explain exactly why it was made."

---

Step 4

Decision Intelligence Studio ⭐

Move slider

Prediction changes

Confidence changes

SHAP changes

Say

"This allows analysts to explore model behavior interactively."

This is your WOW moment.

---

Step 5

Failure Explorer ⭐⭐⭐⭐⭐

Show incorrect prediction

Say

"We intentionally expose incorrect predictions because trustworthy AI requires transparency."

Judges LOVE this.

---

# Slide 8

Performance

Show

Accuracy

Macro F1

Precision

Recall

Inference

Model Size

Mention

"We evaluated multiple baseline models before selecting LightGBM."

Never say

"Our model is 99% accurate."

Explain

WHY.

---

# Slide 9

Engineering

Mention

Docker

FastAPI

SQLite

Logging

GitHub

Documentation

Explain

"This is designed as a deployable ML application."

---

# Slide 10

Future Work

Mention

Real-time IDS

SIEM Integration

Cloud Deployment

Model Drift

LLM Incident Reports

---

# Final Slide

Thank You

Questions

---

# Demo Checklist

Before Presentation

☐ API Running

☐ Dashboard Running

☐ Model Loaded

☐ Sample CSV Ready

☐ Internet Not Required

☐ Browser Cached

☐ Dark Theme Enabled

---

# Demo Dataset

Prepare

Small CSV

20 rows

Fast prediction

Avoid loading millions of rows.

---

# Judge Psychology

Judges ask

Can they build ML?

Can they deploy ML?

Can they explain ML?

Can they defend ML?

Optimize for these questions.

---

# Never Say

"Our model is 99% accurate."

Instead

"Our model balances accuracy, explainability, and deployment feasibility."

---

Never Say

"SHAP tells us everything."

Instead

"SHAP helps interpret feature contributions."

---

Never Say

"This replaces analysts."

Instead

"This assists analysts."

---

# Expected Questions

Why LightGBM?

Why SHAP?

Why CICIDS2017?

Why not Deep Learning?

Why not Transformers?

How would this scale?

How would you deploy this?

How would you retrain?

What are the limitations?

Prepare answers.

---

# Presentation Tips

Speak slowly.

Do not read slides.

Explain architecture.

Explain engineering.

Explain trade-offs.

---

# Backup Plan

If dashboard fails

Use API

If API fails

Use notebook

If notebook fails

Use screenshots

Never stop the presentation.

---

# Success Criteria

Judges should leave thinking

"This team understands Machine Learning engineering, not just Machine Learning."

---

# Final Message

SentinelXAI is not another intrusion detection model.

It is an Explainable AI Decision Intelligence Platform that combines machine learning, explainability, software engineering, and human-centered design into a production-ready cybersecurity solution.