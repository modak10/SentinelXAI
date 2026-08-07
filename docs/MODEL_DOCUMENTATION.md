# MODEL_DOCUMENTATION.md

# SentinelXAI Model Documentation

Version: 1.0

Project: SentinelXAI – Explainable AI Decision Intelligence Platform

Competition: ML Bubble 2026

---

# Purpose

This document describes the complete Machine Learning pipeline used by SentinelXAI.

It includes:

- Problem formulation
- Dataset
- Feature engineering
- Model selection
- Training
- Hyperparameter tuning
- Evaluation
- Explainability
- Inference
- Model persistence
- Versioning
- Limitations
- Future improvements

---

# Machine Learning Problem

## Problem Type

Multi-class Classification

---

## Input

Network Flow Features

---

## Output

Attack Category

Examples

- BENIGN
- Bot
- DDoS
- PortScan
- DoS Hulk
- SSH-Patator
- FTP-Patator
- Web Attack
- Infiltration
- Heartbleed

---

# Business Objective

Help SOC analysts identify malicious traffic quickly while explaining every prediction.

The objective is NOT only classification.

The objective is

Prediction

↓

Confidence

↓

Explanation

↓

Decision Support

---

# Selected Dataset

Dataset

CICIDS2017

Task

Flow Classification

Approximate Records

2.8 Million

Features

78–80

Target

Label

---

# Dataset Split

Training

70%

Validation

15%

Testing

15%

Random State

42

Stratified Split

Yes

---

# Feature Engineering

Allowed

✔ Remove duplicate rows

✔ Remove invalid values

✔ Replace Infinity

✔ Label Encoding

✔ Constant feature removal

✔ Correlation analysis

Forbidden

✘ PCA

✘ Random feature generation

✘ Feature transformations that reduce explainability

---

# Baseline Models

Every project must benchmark multiple algorithms.

Models

1. Logistic Regression

Purpose

Simple interpretable baseline.

---

2. Decision Tree

Purpose

Explainable tree model.

---

3. Random Forest

Purpose

Strong ensemble baseline.

---

4. XGBoost

Purpose

Gradient Boosting benchmark.

---

5. LightGBM

Purpose

Final production model.

---

# Why LightGBM?

Reasons

✔ Excellent performance on tabular data

✔ Fast training

✔ Fast inference

✔ Small model size

✔ CPU optimized

✔ Native TreeSHAP support

✔ Production proven

✔ Low memory usage

LightGBM provides the best balance between

Accuracy

Explainability

Speed

Deployment

---

# Training Pipeline

Raw Dataset

↓

Cleaning

↓

Feature Validation

↓

Train/Test Split

↓

Baseline Models

↓

Hyperparameter Optimization

↓

Train Final LightGBM

↓

Evaluation

↓

SHAP

↓

Save Model

---

# Hyperparameter Optimization

Preferred

Optuna

Alternative

Random Search

Tune

learning_rate

num_leaves

max_depth

feature_fraction

bagging_fraction

min_data_in_leaf

lambda_l1

lambda_l2

Never tune manually.

Store every experiment.

---

# Evaluation Metrics

Primary

Macro F1

Precision

Recall

PR-AUC

ROC-AUC

Matthews Correlation Coefficient

Secondary

Accuracy

Confusion Matrix

Training Time

Inference Time

Model Size

Memory Usage

Never optimize Accuracy alone.

---

# Model Benchmark Table

| Model | Accuracy | Macro F1 | Recall | Precision | Explainability |
|--------|----------|----------|---------|------------|----------------|
| Logistic Regression | TBD | TBD | TBD | TBD | High |
| Decision Tree | TBD | TBD | TBD | TBD | Very High |
| Random Forest | TBD | TBD | TBD | TBD | High |
| XGBoost | TBD | TBD | TBD | TBD | Medium |
| **LightGBM** | TBD | TBD | TBD | TBD | High |

Replace TBD after experimentation.

---

# Explainable AI

Framework

TreeSHAP

Global Explanations

Feature Importance

Summary Plot

Bar Plot

Local Explanations

Waterfall Plot

Decision Plot

Force Plot

Every prediction must include

Prediction

Confidence

Top Features

Feature Contributions

Human Explanation

---

# Confidence Estimation

Every prediction returns

Prediction

Probability

Confidence

Example

Prediction

Bot

Probability

0.98

Confidence

Very High

Low confidence predictions should be flagged.

---

# Human Explanation Engine

Example

Instead of

Feature_12 = +0.42

Display

"High SYN packet count significantly increased the probability of a Botnet attack."

Explanations are deterministic.

Never hallucinate explanations.

---

# Recommendation Engine

Prediction

↓

Risk Mapping

↓

Recommendation

Example

PortScan

↓

Review Firewall Logs

↓

Check Source IP

↓

Inspect IDS History

---

# Failure Explorer

Every ML model makes mistakes.

SentinelXAI intentionally exposes

False Positives

False Negatives

Low Confidence Predictions

Misclassified Samples

The goal is

Trustworthy AI

---

# Model Persistence

Directory

models/

Files

lightgbm_model.pkl

label_encoder.pkl

feature_list.json

hyperparameters.json

training_metadata.json

Use

joblib

for serialization.

---

# Versioning

Every trained model receives

Model Version

Training Date

Dataset Version

Feature Version

Hyperparameters

Evaluation Metrics

Never overwrite models.

---

# Inference Pipeline

Incoming CSV

↓

Validation

↓

Preprocessing

↓

LightGBM

↓

Prediction

↓

Probability

↓

TreeSHAP

↓

Recommendation

↓

API Response

---

# Performance Targets

Training

<10 minutes

Inference

<100 ms

API

<200 ms

Model Size

<10 MB

Memory

<500 MB

CPU

Laptop compatible

GPU

Not required

---

# Logging

Store

Training Start

Training End

Metrics

Hyperparameters

Random Seed

Dataset Version

Model Version

Inference Latency

Prediction Count

---

# Reproducibility

Always save

Random Seed

Dataset Version

Feature List

Library Versions

Hyperparameters

Training Date

Git Commit Hash (optional)

---

# Risks

Class Imbalance

Mitigation

Class weights

Macro F1

Per-class Recall

---

Overfitting

Mitigation

Cross Validation

Early Stopping

Hyperparameter Optimization

---

Concept Drift

Future Work

Periodic Retraining

Drift Detection

---

# Future Improvements

CatBoost Benchmark

Calibration

Ensemble Models

Cross Dataset Validation

Online Learning

Drift Monitoring

Auto Retraining

Model Registry

MLflow Integration

---

# Engineering Principles

Never optimize only for Accuracy.

Always benchmark.

Always explain predictions.

Always expose confidence.

Always save metadata.

Always version models.

Always document experiments.

---

# Definition of Success

✔ Dataset cleaned

✔ Baseline models trained

✔ LightGBM selected

✔ Hyperparameters optimized

✔ SHAP integrated

✔ Confidence estimation implemented

✔ Recommendations generated

✔ Model serialized

✔ Inference under 100 ms

✔ Fully reproducible pipeline