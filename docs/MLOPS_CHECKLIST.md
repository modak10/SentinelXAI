# MLOPS_CHECKLIST.md

# SentinelXAI MLOps Engineering Checklist

Version: 1.0

Project:
SentinelXAI – Explainable AI Decision Intelligence Platform

Competition:
ML Bubble 2026

---

# Purpose

This document defines the Machine Learning Operations (MLOps) standards followed by SentinelXAI.

The goal is to ensure the project is:

- Reproducible
- Maintainable
- Versioned
- Deployable
- Monitorable
- Production Ready

---

# MLOps Lifecycle

```
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
Explainability
    │
    ▼
Model Registry
    │
    ▼
Deployment
    │
    ▼
Monitoring
    │
    ▼
Retraining
```

---

# Repository Standards

```
SentinelXAI/

configs/

data/

docs/

logs/

models/

scripts/

src/

tests/

README.md

CLAUDE.md
```

Never store temporary files in Git.

---

# Dataset Versioning

Every dataset version must record

Dataset Name

Dataset Version

Download Date

Number of Records

Feature Count

Target Classes

Checksum (optional)

Directory

```
data/

raw/

interim/

processed/

external/
```

Never modify files in raw/.

---

# Data Pipeline

Raw

↓

Merge

↓

Cleaning

↓

Validation

↓

Split

↓

Processed Dataset

↓

Training

Every step must be reproducible.

---

# Model Versioning

Every trained model receives

Model Name

Version

Training Date

Dataset Version

Random Seed

Feature Version

Hyperparameters

Evaluation Metrics

Never overwrite models.

Example

```
models/

v1/

lightgbm.pkl

metadata.json

v2/

lightgbm.pkl

metadata.json
```

---

# Experiment Tracking

Record

Algorithm

Hyperparameters

Training Time

Evaluation Metrics

Model Size

Inference Time

Random Seed

Git Commit (optional)

Future

MLflow

Weights & Biases

---

# Model Registry

Required Files

```
lightgbm_model.pkl

label_encoder.pkl

feature_list.json

metadata.json

training_report.json
```

---

# Metadata Example

```
{
    "model_name":"LightGBM",
    "version":"1.0",
    "dataset":"CICIDS2017",
    "accuracy":0.98,
    "macro_f1":0.97,
    "training_date":"..."
}
```

---

# Logging

Every module logs

Training Start

Training End

Prediction Count

Inference Time

Errors

Warnings

Save logs

```
logs/

application.log

training.log

prediction.log

error.log
```

Never use print().

---

# Configuration

Store all configuration

```
configs/

config.yaml

logging.yaml

model.yaml
```

Never hardcode

Paths

Ports

Thresholds

Hyperparameters

---

# Environment Variables

```
.env

HOST

PORT

MODEL_PATH

DATABASE_PATH

LOG_LEVEL
```

Never commit .env.

---

# Reproducibility Checklist

Record

Python Version

Library Versions

Random Seed

Dataset Version

Feature List

Model Version

Training Date

Git Commit (optional)

---

# Training Pipeline

```
Dataset

↓

Cleaning

↓

Feature Validation

↓

Split

↓

Training

↓

Evaluation

↓

Explainability

↓

Save Model
```

Pipeline should be executable with one command.

---

# Inference Pipeline

```
Incoming Data

↓

Validation

↓

Preprocessing

↓

Prediction

↓

Confidence

↓

SHAP

↓

Recommendation

↓

API Response
```

---

# CI/CD (Future)

Git Push

↓

GitHub Actions

↓

Run Tests

↓

Build Docker

↓

Deploy

Current Hackathon Scope

Manual deployment.

---

# Testing

Every release requires

Unit Tests

Integration Tests

API Tests

Prediction Tests

Dashboard Tests

---

# Monitoring (Future)

Monitor

Prediction Count

Latency

CPU

Memory

Model Drift

Data Drift

Error Rate

---

# Drift Detection

Future Feature

Monitor

Feature Distribution

Prediction Distribution

Confidence Changes

Performance Changes

Trigger

Model Retraining

---

# Retraining Strategy

Current

Manual

Future

Scheduled

Monthly

Quarterly

Triggered by drift.

---

# Deployment

Current

Docker

Future

AWS

Azure

Google Cloud

Kubernetes

---

# Backup Strategy

Backup

Models

Database

Logs

Configuration

Frequency

Weekly

---

# Security

Validate Input

Restrict Upload Size

Store Secrets in .env

No Credentials in Git

Sanitize Uploaded CSV

---

# Release Checklist

Before every release

☐ Tests pass

☐ Model saved

☐ Metadata updated

☐ README updated

☐ API tested

☐ Dashboard tested

☐ Docker builds

☐ Logs working

☐ Documentation updated

---

# Production Checklist

☐ Model Versioned

☐ Dataset Versioned

☐ Configuration Externalized

☐ Logging Enabled

☐ API Stable

☐ Dashboard Stable

☐ Docker Ready

☐ Documentation Complete

---

# Engineering Principles

Always

Version datasets

Version models

Document experiments

Log everything

Write reproducible pipelines

Never overwrite artifacts

Never hardcode configuration

Never deploy untested models

---

# Future Improvements

MLflow Integration

DVC for Dataset Versioning

Kubeflow Pipelines

Model Registry

Automatic Retraining

Cloud Deployment

Prometheus Monitoring

Grafana Dashboards

Slack Alerts

Model Drift Detection

Feature Drift Detection

---

# Success Criteria

✔ Reproducible pipeline

✔ Versioned datasets

✔ Versioned models

✔ Logged experiments

✔ Docker deployment

✔ Documented workflow

✔ Production-ready project structure

SentinelXAI follows modern MLOps engineering practices suitable for academic research and production deployment.