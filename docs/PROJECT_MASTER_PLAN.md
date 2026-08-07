# PROJECT MASTER PLAN

# SentinelXAI

## Explainable AI Decision Intelligence Platform for Network Intrusion Detection

---

# Version

1.0

---

# Competition

ML Bubble 2026

---

# Project Vision

SentinelXAI is a production-oriented Explainable Artificial Intelligence platform that assists Security Operations Center (SOC) analysts in identifying, understanding, prioritizing, and investigating cyber threats using Machine Learning.

Unlike conventional Intrusion Detection Systems that stop after predicting whether network traffic is malicious, SentinelXAI transforms Machine Learning into a transparent decision-support system by integrating:

- Explainable AI
- Confidence Estimation
- Risk Assessment
- Interactive Investigation
- Human-in-the-loop Decision Support
- Production Deployment

---

# Mission

Develop an Explainable AI platform that demonstrates the complete Machine Learning engineering lifecycle while improving cybersecurity decision-making.

---

# Objectives

Primary Objectives

✓ Build an accurate intrusion detection model

✓ Explain every prediction

✓ Estimate prediction confidence

✓ Prioritize alerts

✓ Recommend analyst actions

✓ Deploy using modern MLOps practices

Secondary Objectives

✓ Demonstrate software engineering maturity

✓ Showcase Explainable AI

✓ Produce reusable architecture

---

# Problem Statement

Modern organizations generate enormous amounts of network traffic.

Existing IDS solutions suffer from:

- Alert Fatigue

- High False Positives

- Black-box Predictions

- Poor Explainability

- Limited Analyst Trust

- Slow Investigation

Most academic projects optimize only classification accuracy.

SentinelXAI optimizes analyst decision quality.

---

# Target Users

Primary

Security Operations Center Analyst

Secondary

Cybersecurity Researcher

Students

Security Engineers

Educators

---

# Jobs To Be Done

Help SOC analysts

Understand attacks

↓

Understand why

↓

Estimate confidence

↓

Prioritize alerts

↓

Investigate efficiently

---

# Success Criteria

Technical

Macro F1 > Baseline

Low latency

High explainability

Reliable deployment

Product

Interactive dashboard

Decision support

Transparent predictions

Engineering

Modular architecture

Docker deployment

REST API

Documentation

---

# Scope

Included

✓ Intrusion Detection

✓ Explainability

✓ Dashboard

✓ REST API

✓ Docker

✓ Logging

✓ Model Benchmarking

✓ Failure Analysis

Excluded

Live Packet Capture

Cloud Deployment

SIEM Integration

Online Learning

LLM Integration

These remain future work.

---

# Functional Requirements

The platform shall:

Predict attacks

Explain predictions

Estimate confidence

Recommend investigation

Prioritize alerts

Store prediction history

Display analytics

Support batch prediction

---

# Non-functional Requirements

Performance

Inference

<100ms

Reliability

Modular

Maintainable

Documented

Portable

Dockerized

CPU Compatible

---

# Dataset

Primary Dataset

CICIDS2017

Source

Canadian Institute for Cybersecurity

Format

CSV

MachineLearningCSV

Contains

~2.8 Million Flows

80 Features

Multiple Attack Classes

---

# Dataset Workflow

Download

↓

Merge CSV

↓

Cleaning

↓

EDA

↓

Feature Validation

↓

Train/Test Split

↓

Model Training

---

# Technology Stack

Programming

Python 3.11

ML

LightGBM

Scikit-learn

Explainability

SHAP

Frontend

Streamlit

Backend

FastAPI

Database

SQLite

Visualization

Plotly

Deployment

Docker

---

# Project Architecture

User

↓

Dashboard

↓

FastAPI

↓

Prediction Engine

↓

LightGBM

↓

Explainability

↓

Decision Intelligence

↓

Database

---

# Machine Learning Pipeline

Raw CSV

↓

Cleaning

↓

Duplicates Removal

↓

Handle Missing Values

↓

Feature Validation

↓

Train Validation Test Split

↓

Baseline Models

↓

LightGBM

↓

Evaluation

↓

SHAP

↓

Deployment

---

# Baseline Models

Logistic Regression

Decision Tree

Random Forest

XGBoost

Final

LightGBM

---

# Explainability

Global

Feature Importance

Summary Plot

Local

Waterfall Plot

Decision Plot

Human-readable explanation

---

# Decision Intelligence Engine

Outputs

Prediction

Confidence

Risk

Recommendation

Explanation

Priority

---

# Dashboard Modules

Dashboard

Live Prediction

Explainable AI

Decision Intelligence Studio

Failure Explorer

Analytics

About

---

# Backend Modules

Prediction Service

Explainability Service

Recommendation Engine

Database Service

Logging Service

Metrics Service

---

# Database

SQLite

Tables

Prediction History

Model Metadata

Logs

Users (optional)

---

# API

POST /predict

POST /batch_predict

GET /metrics

GET /health

GET /feature-importance

---

# Logging

Application Logs

Prediction Logs

Error Logs

Training Logs

---

# Testing Strategy

Unit Tests

Integration Tests

Model Validation

API Testing

Dashboard Testing

---

# Deployment

Local

Docker

Future

Cloud

Kubernetes

---

# Folder Structure

SentinelXAI/

README.md

CLAUDE.md

docs/

configs/

scripts/

src/

tests/

data/

models/

logs/

---

# Risks

Dataset imbalance

Overfitting

Scope creep

Dashboard complexity

Mitigation

Class weighting

Cross validation

Incremental development

Modular architecture

---

# Timeline

Week 1

Dataset

Cleaning

EDA

Week 2

Training

Benchmarking

Explainability

Week 3

Backend

Dashboard

Week 4

Testing

Docker

Presentation

---

# Deliverables

Source Code

Documentation

Docker Container

GitHub Repository

Presentation

Model

API

Dashboard

---

# Future Enhancements

Real-time IDS

Drift Detection

Active Learning

Federated Learning

Cloud Deployment

SIEM Integration

LLM-powered Reports

Threat Intelligence

---

# Definition of Done

✔ Dataset cleaned

✔ Model benchmarked

✔ LightGBM trained

✔ SHAP integrated

✔ API operational

✔ Dashboard functional

✔ Docker image built

✔ Documentation complete

✔ GitHub repository ready

✔ Presentation prepared

---

# Guiding Principle

SentinelXAI is not another intrusion detection model.

It is an Explainable AI Decision Intelligence Platform designed to improve how humans investigate cyber threats through transparent, trustworthy, and production-ready Machine Learning.