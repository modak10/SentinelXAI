# DEPLOYMENT_GUIDE.md

# SentinelXAI Deployment Guide

Version: 1.0

Project: SentinelXAI – Explainable AI Decision Intelligence Platform

Competition: ML Bubble 2026

---

# Purpose

This document describes how SentinelXAI is deployed locally and how it can be extended to cloud environments.

Deployment goals:

- Reproducible
- Portable
- Easy to setup
- Dockerized
- Production-ready architecture

---

# Deployment Architecture

```
                  User
                    │
                    ▼
          Streamlit Dashboard
                    │
             REST API Request
                    │
                    ▼
               FastAPI Server
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
LightGBM Model   SHAP Engine    SQLite Database
                    │
                    ▼
             Prediction Results
```

---

# Deployment Stack

| Layer | Technology |
|---------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| ML Model | LightGBM |
| Explainability | SHAP |
| Database | SQLite |
| Container | Docker |
| Version Control | GitHub |

---

# Hardware Requirements

Minimum

CPU

2 Core

RAM

8 GB

Storage

5 GB

Recommended

CPU

4–8 Core

RAM

16 GB

Storage

20 GB

GPU

Not Required

---

# Operating Systems

Supported

Windows

Ubuntu

macOS

Docker Desktop

---

# Python Version

Python 3.11+

---

# Environment Setup

Clone repository

```bash
git clone https://github.com/<username>/SentinelXAI.git

cd SentinelXAI
```

---

Create virtual environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Project Structure

```
SentinelXAI/

configs/

data/

logs/

models/

scripts/

src/

tests/

README.md

requirements.txt

Dockerfile

docker-compose.yml
```

---

# Configuration

Configuration files

configs/

config.yaml

logging.yaml

model.yaml

Never modify source code to change configuration.

---

# Environment Variables

Example

.env

```
APP_NAME=SentinelXAI

HOST=0.0.0.0

PORT=8000

MODEL_PATH=models/lightgbm_model.pkl

DATABASE=database/sentinel.db

LOG_LEVEL=INFO
```

---

# Running Locally

Backend

```bash
uvicorn src.api.main:app --reload
```

Swagger

http://localhost:8000/docs

---

Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard URL

http://localhost:8501

---

# Docker Deployment

Build

```bash
docker build -t sentinelxai .
```

Run

```bash
docker run -p 8000:8000 sentinelxai
```

---

# Docker Compose

```bash
docker-compose up --build
```

Expected services

FastAPI

Streamlit

SQLite

---

# Docker Directory

```
docker/

Dockerfile

docker-compose.yml

.dockerignore
```

---

# Dockerfile Workflow

Python Image

↓

Install Requirements

↓

Copy Source

↓

Load Model

↓

Start FastAPI

---

# Deployment Validation

Verify

API running

Dashboard accessible

Model loaded

Database connected

Logs generated

---

# Logging

Store

Application Logs

Prediction Logs

Training Logs

Error Logs

Directory

logs/

---

# Database

SQLite

Purpose

Prediction History

Logs

Metadata

Future

PostgreSQL

---

# Model Files

models/

lightgbm_model.pkl

label_encoder.pkl

feature_list.json

metadata.json

Never store temporary models.

---

# Deployment Checklist

✔ Python installed

✔ Virtual environment created

✔ Dependencies installed

✔ Dataset prepared

✔ Model trained

✔ Model saved

✔ API running

✔ Dashboard running

✔ Logs generated

✔ Docker image built

---

# CI/CD (Future)

GitHub Actions

Pipeline

Push

↓

Run Tests

↓

Build Docker

↓

Deploy

---

# Cloud Deployment (Future)

AWS

Azure

Google Cloud

Render

Railway

Fly.io

---

# Monitoring

Future

Prometheus

Grafana

MLflow

Drift Detection

Health Monitoring

---

# Backup Strategy

Backup

Models

Database

Configuration

Logs

Weekly

---

# Security

Validate uploads

Restrict file size

Sanitize inputs

Store secrets in .env

Never commit secrets

---

# Performance Targets

API Response

<200 ms

Prediction

<100 ms

Dashboard Load

<3 seconds

Docker Startup

<30 seconds

---

# Common Issues

Problem

Model not found

Solution

Verify MODEL_PATH

---

Problem

Dashboard not loading

Solution

Check Streamlit port

---

Problem

API unavailable

Solution

Check FastAPI logs

---

Problem

Import errors

Solution

Install requirements again

---

# Deployment Acceptance Criteria

✔ Backend operational

✔ Dashboard operational

✔ API documented

✔ Docker works

✔ Configuration externalized

✔ Logs generated

✔ Model loads successfully

✔ Ready for demonstration

---

# Future Improvements

Kubernetes

NGINX Reverse Proxy

HTTPS

Redis Cache

Load Balancer

Cloud Deployment

Auto Scaling

Continuous Deployment
