# TASKS.md

# SentinelXAI Development Task Board

Project

SentinelXAI – Explainable AI Decision Intelligence Platform

Competition

ML Bubble 2026

---

# Project Progress

Overall Progress

**~80%** — all engineering layers (Phases 1–13) are implemented and covered by a
passing test suite (145 tests, `ruff` clean). Remaining work is documentation
polish, a presentation, and GitHub repository polish (Phases 15–17). The trained
model artifact is git-ignored and not committed; the app degrades gracefully until
`make data` + `scripts/train_final_lightgbm.py` are run.

Legend

⬜ Not Started

🟨 In Progress

🟩 Completed

Status by Phase

- 🟩 Phase 1 — Project Initialization
- 🟩 Phase 2 — Dataset Engineering
- 🟩 Phase 3 — Exploratory Data Analysis
- 🟩 Phase 4 — Feature Engineering
- 🟩 Phase 5 — Machine Learning Benchmark
- 🟩 Phase 6 — Hyperparameter Optimization
- 🟩 Phase 7 — Explainable AI (SHAP)
- 🟩 Phase 8 — Decision Intelligence Engine
- 🟩 Phase 9 — FastAPI Backend
- 🟩 Phase 10 — Streamlit Dashboard
- 🟩 Phase 11 — Database (SQLite)
- 🟩 Phase 12 — Logging
- 🟩 Phase 13 — Docker
- 🟨 Phase 14 — Testing (unit + API automated; dashboard/integration partial)
- 🟨 Phase 15 — Documentation (README updated; other docs pending)
- ⬜ Phase 16 — GitHub Polish
- ⬜ Phase 17 — Presentation
- ⬜ Stretch Goals

---

# Phase 1 — Project Initialization

## Repository

- [x] Create GitHub Repository
- [x] Configure Git
- [x] Create .gitignore
- [x] Create README.md
- [x] Create LICENSE

---

## Python Environment

- [x] Create Virtual Environment
- [x] Install Python 3.11+
- [x] Install dependencies
- [x] Freeze requirements.txt

---

## Project Structure

- [x] Create docs/
- [x] Create configs/
- [x] Create src/
- [x] Create scripts/
- [x] Create data/
- [x] Create models/
- [x] Create logs/
- [x] Create notebooks/
- [x] Create tests/

---

## Configuration

- [x] config.yaml
- [x] logging.yaml
- [x] .env.example
- [x] configs/decision.yaml (Phase 8 policy)

---

# Phase 2 — Dataset Engineering

## Dataset Download

- [x] Download MachineLearningCSV.zip
- [x] Extract dataset
- [x] Verify all CSV files

---

## Data Merge

- [x] Merge all CSV files
- [x] Verify columns
- [x] Verify labels

---

## Data Cleaning

- [x] Remove duplicate rows
- [x] Replace Infinity values
- [x] Handle missing values
- [x] Validate datatypes
- [x] Remove invalid rows

---

## Dataset Validation

- [x] Verify label distribution
- [x] Verify feature count
- [x] Verify target column
- [x] Save merged dataset
- [x] Generate preprocessing report

---

## Dataset Split

- [x] Train Set (70%)
- [x] Validation Set (15%)
- [x] Test Set (15%)
- [x] Save processed datasets

---

# Phase 3 — Exploratory Data Analysis

## Dataset Overview

- [x] Dataset shape
- [x] Data types
- [x] Summary statistics

---

## Visualizations

- [x] Class distribution
- [x] Missing value heatmap
- [x] Correlation matrix
- [x] Feature histograms
- [x] Boxplots
- [x] Pair plots (selected features)

---

## Analysis

- [x] Identify imbalance
- [x] Identify outliers
- [x] Identify redundant features
- [x] Generate EDA report

---

# Phase 4 — Feature Engineering

## Feature Validation

- [x] Constant feature removal
- [x] Duplicate feature check
- [x] Correlation analysis

---

## Feature Selection

- [x] Select final feature list
- [x] Save feature names

---

## Label Processing

- [x] Encode labels
- [x] Save label encoder

---

# Phase 5 — Machine Learning Benchmark

## Logistic Regression

- [x] Train
- [x] Evaluate
- [x] Save metrics

---

## Decision Tree

- [x] Train
- [x] Evaluate

---

## Random Forest

- [x] Train
- [x] Evaluate

---

## XGBoost

- [x] Train
- [x] Evaluate

---

## LightGBM

- [x] Train
- [x] Evaluate

---

## Benchmark Report

- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] Macro F1
- [x] ROC-AUC
- [x] PR-AUC
- [x] MCC
- [x] Training Time
- [x] Inference Time

---

# Phase 6 — Hyperparameter Optimization

## Optimization

- [x] Configure Optuna
- [x] Run optimization
- [x] Save best parameters
- [x] Retrain final model

---

## Model Persistence

- [x] Save model
- [x] Save metadata
- [x] Save feature list
- [x] Save label encoder

---

# Phase 7 — Explainable AI

## SHAP

- [x] Global Feature Importance
- [x] Summary Plot
- [x] Waterfall Plot
- [x] Decision Plot
- [x] Force Plot

---

## Human Explanation

- [x] Generate explanation templates
- [x] Map SHAP to readable text

---

# Phase 8 — Decision Intelligence Engine

## Confidence

- [x] Probability calculation
- [x] Confidence levels

---

## Risk Engine

- [x] Risk mapping
- [x] Severity mapping

---

## Recommendation Engine

- [x] Rule mapping
- [x] Attack recommendations

---

## Failure Explorer

- [x] Misclassified samples
- [x] Low confidence predictions
- [x] Error explanations

---

## Decision Simulator

- [x] Interactive sliders
- [x] Live prediction updates
- [x] Live SHAP updates

---

# Phase 9 — FastAPI Backend

## API

- [x] Create FastAPI project
- [x] Configure routes
- [x] Configure middleware

---

## Endpoints

- [x] POST /predict
- [x] POST /batch_predict
- [x] POST /upload
- [x] GET /health
- [x] GET /metrics
- [x] GET /model
- [x] GET /feature-importance
- [ ] GET /history (logged to SQLite; endpoint not yet exposed)

---

## Validation

- [x] Pydantic models
- [x] Error handling
- [x] Logging

---

# Phase 10 — Streamlit Dashboard

## Dashboard

- [x] Home
- [x] Navigation
- [x] Status cards

---

## Prediction

- [x] CSV upload
- [x] Prediction page

---

## Explainability

- [x] SHAP visualization
- [x] Human explanation

---

## Decision Intelligence Studio

- [x] Interactive sliders
- [x] Live prediction
- [x] Confidence
- [x] Recommendations

---

## Failure Explorer

- [x] Error analysis
- [x] Misclassified samples

---

## Analytics

- [ ] Confusion Matrix (static placeholders only)
- [ ] ROC Curve
- [ ] PR Curve
- [ ] Model Comparison

---

## About

- [x] Project information
- [x] Architecture
- [x] Tech stack

---

# Phase 11 — Database

## SQLite

- [x] Create database
- [x] Prediction table
- [x] Logs table
- [x] Metadata table

---

# Phase 12 — Logging

- [x] Application log
- [x] Training log
- [x] Prediction log
- [x] Error log

---

# Phase 13 — Docker

- [x] Dockerfile
- [x] docker-compose.yml
- [x] Build image
- [ ] Run container (validated via config; full run needs real artifacts)

---

# Phase 14 — Testing

## Unit Tests

- [x] Preprocessing
- [x] Training
- [x] Evaluation
- [x] SHAP
- [x] Decision Engine

---

## API Tests

- [x] Prediction endpoint
- [x] Upload endpoint
- [x] Health endpoint

---

## Dashboard Tests

- [ ] Page loading (manual smoke test passed; no automated test)
- [ ] CSV upload
- [ ] Charts

---

## Integration Tests

- [ ] End-to-end prediction
- [ ] Database logging
- [ ] API integration

---

# Phase 15 — Documentation

- [x] Update README
- [ ] Update Architecture
- [ ] Update API docs
- [ ] Update Dataset Guide
- [ ] Update Model Documentation

---

# Phase 16 — GitHub

- [ ] Push source code
- [ ] Add screenshots
- [ ] Add badges
- [ ] Add documentation
- [ ] Add LICENSE

---

# Phase 17 — Presentation

## PPT

- [ ] Problem
- [ ] Solution
- [ ] Dataset
- [ ] ML Pipeline
- [ ] Architecture
- [ ] Demo
- [ ] Results
- [ ] Future Work

---

## Demo

- [ ] Sample CSV ready
- [ ] Dashboard working
- [ ] API working
- [ ] SHAP working
- [ ] Decision Studio working
- [ ] Failure Explorer working

---

## Q&A

- [ ] Practice technical questions
- [ ] Practice architecture explanation
- [ ] Practice ML explanation
- [ ] Practice deployment explanation

---

# Stretch Goals

- [ ] MLflow
- [ ] Drift Detection
- [ ] Cross-dataset Validation
- [ ] UNSW-NB15 Evaluation
- [ ] Docker Compose
- [ ] GitHub Actions
- [ ] Model Monitoring
- [ ] Cloud Deployment
- [ ] SIEM Integration
- [ ] LLM Incident Reports

---

# Definition of Done

Project is complete when

- [x] Dataset cleaned
- [x] EDA completed
- [x] Benchmark completed
- [x] LightGBM selected
- [x] SHAP integrated
- [x] Decision Engine complete
- [x] FastAPI operational
- [x] Streamlit dashboard operational
- [x] Docker deployment working
- [x] Tests passing
- [ ] Documentation complete
- [ ] GitHub repository polished
- [ ] Presentation rehearsed
- [ ] Ready for ML Bubble 2026
