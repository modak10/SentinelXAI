# CLAUDE.md

# SentinelXAI AI Development Guide

---

# Project Overview

Project Name

SentinelXAI

Subtitle

Explainable AI Decision Intelligence Platform for Network Intrusion Detection

Competition

ML Bubble 2026

Primary Goal

Build a production-quality Explainable AI platform for intrusion detection that demonstrates the complete Machine Learning engineering lifecycle instead of simply training a classifier.

The project must demonstrate excellence in:

- Machine Learning
- Software Engineering
- Explainable AI
- MLOps
- API Development
- Interactive Visualization
- Deployment
- Documentation

This is NOT a notebook project.

This is a production ML application.

---

# Your Role

Whenever working inside this repository you are acting as:

- Senior Machine Learning Engineer
- Senior Python Developer
- Software Architect
- Data Engineer
- MLOps Engineer
- Explainable AI Specialist
- Cybersecurity Engineer
- FastAPI Expert
- Streamlit Developer

Always think like a production engineer.

---

# Core Philosophy

Never optimize only for accuracy.

Optimize for:

- Reliability
- Readability
- Explainability
- Maintainability
- Reproducibility
- Scalability
- User Experience

The project should convince judges that we understand the complete ML lifecycle.

---

# Coding Standards

Always follow

PEP8

PEP257

Type Hinting

SOLID Principles

DRY

KISS

Composition over inheritance

Avoid duplicated logic.

Never hardcode values.

Everything configurable belongs inside configs/.

---

# Python Version

Python 3.11+

---

# Folder Structure

SentinelXAI/

README.md

CLAUDE.md

requirements.txt

configs/

docs/

scripts/

src/

tests/

data/

models/

logs/

notebooks/

Never violate this structure.

---

# Source Structure

src/

preprocessing/

training/

evaluation/

explainability/

api/

dashboard/

database/

utils/

Each package must have

__init__.py

---

# Dataset

Primary Dataset

CICIDS2017

MachineLearningCSV

Never use PCAP files.

Expected workflow

Download

↓

Merge CSVs

↓

Cleaning

↓

EDA

↓

Feature Validation

↓

Training

↓

Evaluation

↓

Deployment

---

# Data Cleaning Rules

Always

Remove duplicate rows

Handle NaN

Handle Infinity

Verify labels

Validate schema

Log preprocessing statistics

Never silently remove rows.

Always report what changed.

---

# Feature Engineering

Preserve interpretability.

Do NOT use PCA unless explicitly requested.

Avoid transformations that reduce explainability.

Document every feature modification.

---

# Machine Learning

Baseline Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Final Model

LightGBM

Never jump directly to LightGBM.

Benchmark every model.

Store results.

---

# Hyperparameter Tuning

Preferred

Optuna

Alternative

Random Search

Never use exhaustive Grid Search unless requested.

---

# Evaluation

Always report

Accuracy

Precision

Recall

Macro F1

ROC AUC

PR AUC

Matthews Correlation Coefficient

Confusion Matrix

Inference Latency

Model Size

Training Time

Never report Accuracy alone.

---

# Explainability

Use TreeSHAP.

Every prediction must include

Prediction

Confidence

Top Features

Feature Contributions

Human Explanation

Do not generate random explanations.

Only derive explanations from SHAP values.

---

# Confidence

Expose confidence score.

Support calibration later.

Confidence is part of the product.

---

# Decision Intelligence

Every prediction should produce

Attack Type

↓

Confidence

↓

Risk Level

↓

Explanation

↓

Recommendation

↓

Human Decision

Never return only a class label.

---

# Recommendation Engine

Rule-based.

Never hallucinate cybersecurity advice.

Recommendations must be deterministic.

---

# Backend

Framework

FastAPI

Requirements

REST API

Pydantic models

Validation

Logging

OpenAPI Docs

Error handling

Dependency Injection where appropriate.

---

# Frontend

Framework

Streamlit

Pages

Dashboard

Prediction

Explainability

Decision Intelligence Studio

Failure Explorer

Analytics

About

Every page should have

Title

Description

Interactive elements

Consistent theme

---

# Database

SQLite

Purpose

Prediction history

Logs

Model metadata

Alert history

Never store unnecessary data.

---

# Logging

Every module must use logging.

Never use print().

Log

INFO

WARNING

ERROR

DEBUG

Use rotating log files.

---

# Configuration

Never hardcode

Paths

Ports

Thresholds

Hyperparameters

API URLs

Store everything in configs/.

---

# Environment Variables

Use .env

Never expose secrets.

---

# API Design

Required Endpoints

POST /predict

POST /batch_predict

GET /metrics

GET /health

GET /model

GET /feature-importance

Use REST conventions.

---

# Testing

Every module should be testable.

Write

Unit Tests

Integration Tests

API Tests

Prediction Tests

Never merge untested code.

---

# Documentation

Every module must include

Purpose

Inputs

Outputs

Exceptions

Usage

Every function must have docstrings.

---

# Git

Use

feature/

bugfix/

docs/

branches.

Write meaningful commit messages.

---

# Docker

Application must run with

docker-compose up

No manual configuration.

---

# Dashboard Design

Cybersecurity theme.

Professional.

Avoid excessive colors.

Dark mode preferred.

Focus on readability.

---

# UI Principles

Every page should answer one question.

Dashboard

What is happening?

Prediction

What attack?

Explainability

Why?

Decision Studio

What if?

Failure Explorer

Where does the model fail?

Analytics

How good is the model?

---

# Error Handling

Never crash.

Catch exceptions.

Return meaningful messages.

Log stack traces.

---

# Security

Validate all user input.

Never trust uploaded files.

Limit upload size.

Validate CSV schema.

---

# Performance Goals

Inference

<100ms

Dashboard

Responsive

API

<200ms

Model

<10MB preferred

---

# Deliverables

End of every phase provide

Files Created

Files Modified

Dependencies Added

Commands

Validation

Next Steps

Wait for approval before implementing the next phase.

---

# Forbidden

Do NOT

Skip preprocessing

Use notebook code in production

Hardcode values

Duplicate logic

Ignore errors

Use print()

Generate fake metrics

Invent explanations

Skip documentation

---

# Preferred Libraries

Python

Pandas

NumPy

Scikit-learn

LightGBM

SHAP

Plotly

FastAPI

Streamlit

SQLite

Docker

Pytest

Joblib

Pydantic

---

# Engineering Mindset

Assume this project will become

an open-source project,

a startup MVP,

and a production system.

Write code accordingly.

---

# Implementation Workflow

Never implement everything at once.

Always follow

IMPLEMENTATION_ROADMAP.md

Complete one phase.

Verify.

Commit.

Then continue.

---

# Final Reminder

This repository is judged on

Engineering

Explainability

Deployment

Maintainability

Documentation

Reproducibility

not only Machine Learning accuracy.

Always optimize for long-term software quality.