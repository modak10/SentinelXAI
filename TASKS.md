# TASKS.md

# SentinelXAI Development Task Board

Project

SentinelXAI – Explainable AI Decision Intelligence Platform

Competition

ML Bubble 2026

---

# Project Progress

Overall Progress

0%

Legend

⬜ Not Started

🟨 In Progress

🟩 Completed

---

# Phase 1 — Project Initialization

## Repository

- [ ] Create GitHub Repository
- [ ] Configure Git
- [ ] Create .gitignore
- [ ] Create README.md
- [ ] Create LICENSE

---

## Python Environment

- [ ] Create Virtual Environment
- [ ] Install Python 3.11+
- [ ] Install dependencies
- [ ] Freeze requirements.txt

---

## Project Structure

- [ ] Create docs/
- [ ] Create configs/
- [ ] Create src/
- [ ] Create scripts/
- [ ] Create data/
- [ ] Create models/
- [ ] Create logs/
- [ ] Create notebooks/
- [ ] Create tests/

---

## Configuration

- [ ] config.yaml
- [ ] logging.yaml
- [ ] .env.example

---

# Phase 2 — Dataset Engineering

## Dataset Download

- [ ] Download MachineLearningCSV.zip
- [ ] Extract dataset
- [ ] Verify all CSV files

---

## Data Merge

- [ ] Merge all CSV files
- [ ] Verify columns
- [ ] Verify labels

---

## Data Cleaning

- [ ] Remove duplicate rows
- [ ] Replace Infinity values
- [ ] Handle missing values
- [ ] Validate datatypes
- [ ] Remove invalid rows

---

## Dataset Validation

- [ ] Verify label distribution
- [ ] Verify feature count
- [ ] Verify target column
- [ ] Save merged dataset
- [ ] Generate preprocessing report

---

## Dataset Split

- [ ] Train Set (70%)
- [ ] Validation Set (15%)
- [ ] Test Set (15%)
- [ ] Save processed datasets

---

# Phase 3 — Exploratory Data Analysis

## Dataset Overview

- [ ] Dataset shape
- [ ] Data types
- [ ] Summary statistics

---

## Visualizations

- [ ] Class distribution
- [ ] Missing value heatmap
- [ ] Correlation matrix
- [ ] Feature histograms
- [ ] Boxplots
- [ ] Pair plots (selected features)

---

## Analysis

- [ ] Identify imbalance
- [ ] Identify outliers
- [ ] Identify redundant features
- [ ] Generate EDA report

---

# Phase 4 — Feature Engineering

## Feature Validation

- [ ] Constant feature removal
- [ ] Duplicate feature check
- [ ] Correlation analysis

---

## Feature Selection

- [ ] Select final feature list
- [ ] Save feature names

---

## Label Processing

- [ ] Encode labels
- [ ] Save label encoder

---

# Phase 5 — Machine Learning Benchmark

## Logistic Regression

- [ ] Train
- [ ] Evaluate
- [ ] Save metrics

---

## Decision Tree

- [ ] Train
- [ ] Evaluate

---

## Random Forest

- [ ] Train
- [ ] Evaluate

---

## XGBoost

- [ ] Train
- [ ] Evaluate

---

## LightGBM

- [ ] Train
- [ ] Evaluate

---

## Benchmark Report

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Macro F1
- [ ] ROC-AUC
- [ ] PR-AUC
- [ ] MCC
- [ ] Training Time
- [ ] Inference Time

---

# Phase 6 — Hyperparameter Optimization

## Optimization

- [ ] Configure Optuna
- [ ] Run optimization
- [ ] Save best parameters
- [ ] Retrain final model

---

## Model Persistence

- [ ] Save model
- [ ] Save metadata
- [ ] Save feature list
- [ ] Save label encoder

---

# Phase 7 — Explainable AI

## SHAP

- [ ] Global Feature Importance
- [ ] Summary Plot
- [ ] Waterfall Plot
- [ ] Decision Plot
- [ ] Force Plot

---

## Human Explanation

- [ ] Generate explanation templates
- [ ] Map SHAP to readable text

---

# Phase 8 — Decision Intelligence Engine

## Confidence

- [ ] Probability calculation
- [ ] Confidence levels

---

## Risk Engine

- [ ] Risk mapping
- [ ] Severity mapping

---

## Recommendation Engine

- [ ] Rule mapping
- [ ] Attack recommendations

---

## Failure Explorer

- [ ] Misclassified samples
- [ ] Low confidence predictions
- [ ] Error explanations

---

## Decision Simulator

- [ ] Interactive sliders
- [ ] Live prediction updates
- [ ] Live SHAP updates

---

# Phase 9 — FastAPI Backend

## API

- [ ] Create FastAPI project
- [ ] Configure routes
- [ ] Configure middleware

---

## Endpoints

- [ ] POST /predict
- [ ] POST /batch_predict
- [ ] POST /upload
- [ ] GET /health
- [ ] GET /metrics
- [ ] GET /model
- [ ] GET /feature-importance
- [ ] GET /history

---

## Validation

- [ ] Pydantic models
- [ ] Error handling
- [ ] Logging

---

# Phase 10 — Streamlit Dashboard

## Dashboard

- [ ] Home
- [ ] Navigation
- [ ] Status cards

---

## Prediction

- [ ] CSV upload
- [ ] Prediction page

---

## Explainability

- [ ] SHAP visualization
- [ ] Human explanation

---

## Decision Intelligence Studio

- [ ] Interactive sliders
- [ ] Live prediction
- [ ] Confidence
- [ ] Recommendations

---

## Failure Explorer

- [ ] Error analysis
- [ ] Misclassified samples

---

## Analytics

- [ ] Confusion Matrix
- [ ] ROC Curve
- [ ] PR Curve
- [ ] Model Comparison

---

## About

- [ ] Project information
- [ ] Architecture
- [ ] Tech stack

---

# Phase 11 — Database

## SQLite

- [ ] Create database
- [ ] Prediction table
- [ ] Logs table
- [ ] Metadata table

---

# Phase 12 — Logging

- [ ] Application log
- [ ] Training log
- [ ] Prediction log
- [ ] Error log

---

# Phase 13 — Docker

- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] Build image
- [ ] Run container

---

# Phase 14 — Testing

## Unit Tests

- [ ] Preprocessing
- [ ] Training
- [ ] Evaluation
- [ ] SHAP
- [ ] Decision Engine

---

## API Tests

- [ ] Prediction endpoint
- [ ] Upload endpoint
- [ ] Health endpoint

---

## Dashboard Tests

- [ ] Page loading
- [ ] CSV upload
- [ ] Charts

---

## Integration Tests

- [ ] End-to-end prediction
- [ ] Database logging
- [ ] API integration

---

# Phase 15 — Documentation

- [ ] Update README
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

- [ ] Dataset cleaned
- [ ] EDA completed
- [ ] Benchmark completed
- [ ] LightGBM selected
- [ ] SHAP integrated
- [ ] Decision Engine complete
- [ ] FastAPI operational
- [ ] Streamlit dashboard operational
- [ ] Docker deployment working
- [ ] Tests passing
- [ ] Documentation complete
- [ ] GitHub repository polished
- [ ] Presentation rehearsed
- [ ] Ready for ML Bubble 2026