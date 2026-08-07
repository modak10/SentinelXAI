# TESTING_CHECKLIST.md

# SentinelXAI Testing Checklist

Version: 1.0

Project:
SentinelXAI – Explainable AI Decision Intelligence Platform

Competition:
ML Bubble 2026

---

# Purpose

This document defines the testing strategy for SentinelXAI.

The objective is to verify:

- Data Quality
- Machine Learning
- Explainability
- Backend
- Dashboard
- Deployment
- User Experience

Every feature must pass testing before being merged.

---

# Testing Pyramid

```
                End-to-End Tests
                       ▲
               Integration Tests
                       ▲
                 Unit Tests
```

Recommended

- Unit Tests: 70%
- Integration Tests: 20%
- End-to-End Tests: 10%

---

# Testing Framework

Python

pytest

Coverage

pytest-cov

Mocking

pytest-mock

API

FastAPI TestClient

---

# Test Directory

```
tests/

test_preprocessing.py

test_training.py

test_evaluation.py

test_shap.py

test_decision_engine.py

test_api.py

test_dashboard.py

test_database.py

```

---

# Phase 1

Project Setup

Checklist

☐ Repository cloned

☐ Virtual environment created

☐ Dependencies installed

☐ Configuration loaded

☐ Logging works

☐ Project starts successfully

---

# Phase 2

Dataset Testing

Verify

☐ Dataset downloaded

☐ CSV files detected

☐ Correct number of files

☐ Schema consistent

☐ Labels present

☐ No missing columns

---

# Data Cleaning Tests

Verify

☐ Duplicate rows removed

☐ NaN removed

☐ Infinity removed

☐ Invalid rows logged

☐ Output dataset generated

---

# Dataset Split

Verify

☐ Train generated

☐ Validation generated

☐ Test generated

☐ Stratified split

☐ No overlap

---

# Model Training

Verify

☐ Logistic Regression trains

☐ Decision Tree trains

☐ Random Forest trains

☐ XGBoost trains

☐ LightGBM trains

---

# Hyperparameter Search

Verify

☐ Search completes

☐ Best parameters saved

☐ Metrics improve

☐ Model reproducible

---

# Evaluation

Verify

☐ Accuracy calculated

☐ Precision calculated

☐ Recall calculated

☐ Macro F1 calculated

☐ ROC-AUC calculated

☐ PR-AUC calculated

☐ MCC calculated

---

# Explainability

Verify

☐ SHAP initializes

☐ Summary Plot generated

☐ Waterfall Plot generated

☐ Feature Importance generated

☐ Human explanation generated

---

# Confidence

Verify

☐ Probability returned

☐ Confidence displayed

☐ Low confidence flagged

---

# Recommendation Engine

Verify

Prediction

↓

Recommendation

Exists

Every class should return recommendations.

---

# Failure Explorer

Verify

☐ False positives displayed

☐ False negatives displayed

☐ Low confidence highlighted

☐ SHAP explanation shown

---

# FastAPI

Verify

POST /predict

☐ Returns 200

☐ Invalid request returns 400

☐ Validation works

☐ JSON valid

---

POST /batch_predict

☐ CSV upload

☐ Multiple predictions

☐ Handles invalid CSV

---

GET /health

☐ Status OK

☐ Model loaded

☐ Database connected

---

GET /metrics

☐ Metrics returned

☐ JSON valid

---

GET /feature-importance

☐ Feature list returned

☐ Importance sorted

---

# Dashboard

Verify

☐ Dashboard loads

☐ Sidebar works

☐ Upload works

☐ Prediction works

☐ Explainability works

☐ Analytics loads

☐ Failure Explorer loads

☐ Decision Studio interactive

---

# Database

Verify

☐ SQLite created

☐ Predictions saved

☐ Logs saved

☐ Metadata stored

---

# Logging

Verify

Application Log

Prediction Log

Training Log

Error Log

No crashes.

---

# Docker

Verify

☐ Docker build successful

☐ Docker Compose works

☐ API starts

☐ Dashboard starts

☐ Model loads

---

# Performance Tests

Target

Prediction

<100 ms

API

<200 ms

Dashboard

<3 sec

Model Size

<10 MB

---

# Stress Testing

Verify

100 Predictions

500 Predictions

1000 Predictions

API remains responsive.

---

# Error Handling

Upload invalid CSV

↓

Proper error

No crash

---

Upload empty CSV

↓

Proper error

---

Missing Features

↓

Validation error

---

Corrupted Model

↓

Graceful failure

---

# Security Tests

Verify

☐ Invalid uploads rejected

☐ Empty files rejected

☐ Wrong file type rejected

☐ SQL Injection impossible

☐ Path traversal impossible

---

# User Acceptance Test

SOC Analyst Workflow

Open Dashboard

↓

Upload CSV

↓

Predict

↓

Understand explanation

↓

Review recommendation

↓

Investigate attack

Workflow should complete without confusion.

---

# Regression Testing

Before every commit

Verify

☐ Existing API works

☐ Dashboard works

☐ SHAP works

☐ Prediction works

☐ Model loads

---

# Code Quality

Run

black

ruff

pytest

mypy

Before merging.

---

# Documentation Validation

Verify

README updated

API documented

Architecture updated

Model documentation updated

---

# Final Release Checklist

☐ Dataset prepared

☐ Model benchmarked

☐ SHAP integrated

☐ API operational

☐ Dashboard operational

☐ Docker works

☐ Logs generated

☐ Tests pass

☐ Documentation complete

☐ Ready for presentation

---

# Success Criteria

✔ All unit tests pass

✔ All API tests pass

✔ Dashboard functional

✔ Model reproducible

✔ Docker deploys successfully

✔ End-to-end workflow validated

SentinelXAI is ready for ML Bubble 2026.