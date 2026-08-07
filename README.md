# SentinelXAI
### Explainable AI Decision Intelligence Platform for Network Intrusion Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-orange)
![Docker](https://img.shields.io/badge/Docker-Deployment-blue)

---

# Project Status

**Milestone 1 (Project Setup + Dataset Engineering + EDA): complete and verified
against the real dataset.** Milestones 2-4 (ML training, SHAP/API, dashboard/Docker)
are not yet implemented — see `docs/IMPLEMENTATION_ROADMAP.md` status below and
`docs/DATASET_GUIDE.md` for full pipeline detail and verified numbers.

- 2,830,743 raw rows across 8 CICIDS2017 files merged, cleaned, validated
- 307,078 duplicate rows and 2,867 NaN/±Infinity-affected rows removed and reported
- 15-class taxonomy validated; all classes preserved through a leakage-aware
  stratified 70/15/15 split (rare classes documented, never dropped)
- 25+ unit tests passing, `ruff` clean
- Reproducible end-to-end in ~4.5 minutes via `python scripts/build_dataset.py`

---

# Overview

SentinelXAI is an Explainable AI-powered Decision Intelligence Platform designed to assist Security Operations Center (SOC) analysts in detecting, understanding, and investigating cyber threats.

Unlike traditional Machine Learning Intrusion Detection Systems (IDS), SentinelXAI does not stop at predicting whether traffic is malicious. It provides confidence-aware predictions, explainable AI, attack prioritization, investigation recommendations, and an interactive decision-support interface that enables analysts to understand why a prediction was made and how they should respond.

The project demonstrates the complete Machine Learning lifecycle including:

- Data Engineering
- Machine Learning
- Explainable AI (XAI)
- MLOps
- REST APIs
- Interactive Dashboard
- Docker Deployment
- Production Engineering

---

# Project Vision

Modern Intrusion Detection Systems often behave as black boxes.

SentinelXAI transforms intrusion detection into a transparent decision-support platform by combining:

- Accurate Machine Learning
- Explainability
- Human-centered AI
- Production Deployment
- Interactive Investigation

Instead of replacing cybersecurity analysts, SentinelXAI augments their decision-making process.

---

# Key Features

## Machine Learning

- Multi-class Intrusion Detection
- LightGBM Classification
- Confidence Estimation
- Attack Severity Classification

---

## Explainable AI

- SHAP Explanations
- Global Feature Importance
- Local Prediction Explanation
- Human-readable AI Explanations

---

## Decision Intelligence

- Alert Prioritization
- Investigation Recommendations
- Interactive Decision Intelligence Studio
- Failure Explorer

---

## Engineering

- FastAPI REST API
- Streamlit Dashboard
- Docker Deployment
- SQLite Logging
- Modular Architecture

---

# Project Architecture

```
                  User

                    │

                    ▼

         Streamlit Dashboard

                    │

                    ▼

             FastAPI Backend

                    │

    ┌───────────────┼────────────────┐

    ▼               ▼                ▼

Prediction      SHAP Engine      Database

    │

    ▼

 LightGBM Model

    │

    ▼

Data Preprocessing

    │

    ▼

 CICIDS2017 Dataset
```

---

# Technology Stack

| Layer | Technology |
|--------|------------|
| Programming | Python 3.11 |
| Machine Learning | LightGBM |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Visualization | Plotly |
| Deployment | Docker |
| Version Control | Git |

---

# Dataset

Primary Dataset

**CICIDS2017**

Contains:

- 2.8 Million Network Flows
- 80 Network Features
- Multiple Attack Categories
- Realistic Enterprise Traffic

Directory

```
data/raw/

MachineLearningCSV/

Monday.csv

Tuesday.csv

Wednesday.csv

Thursday.csv

Friday.csv
```

---

# Repository Structure

```
SentinelXAI/

│

├── README.md

├── CLAUDE.md

├── requirements.txt

├── docker-compose.yml

├── LICENSE

├── TODO.md

│

├── docs/

│     PROJECT_MASTER_PLAN.md

│     IMPLEMENTATION_ROADMAP.md

│     ARCHITECTURE.md

│     API_SPECIFICATION.md

│     DATASET_GUIDE.md

│     MODEL_DOCUMENTATION.md

│     DEPLOYMENT_GUIDE.md

│

├── configs/

├── scripts/

├── src/

│      preprocessing/

│      training/

│      evaluation/

│      explainability/

│      api/

│      dashboard/

│      database/

│      utils/

│

├── data/

├── models/

├── notebooks/

├── tests/

└── logs/
```

---

# Machine Learning Pipeline

```
Network Flow

↓

Cleaning

↓

Feature Engineering

↓

Train / Validation / Test Split

↓

Baseline Models

↓

LightGBM

↓

Prediction

↓

TreeSHAP

↓

Decision Intelligence

↓

Dashboard
```

---

# Dashboard Modules

- Dashboard
- Live Prediction
- Explainable AI
- Decision Intelligence Studio
- Failure Explorer
- Model Analytics
- About

---

# API

Endpoints

```
POST /predict

POST /batch_predict

GET /metrics

GET /health

GET /model

GET /feature-importance
```

---

# Installation

Clone

```bash
git clone https://github.com/yourusername/SentinelXAI.git

cd SentinelXAI
```

Create environment (this repo's venv folder is named `venv/`, not `.venv/`)

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install (Milestone 1 dependencies only — ML/API/UI packages are added in later
milestones, see requirements.txt)

```bash
pip install -r requirements-dev.txt
pip install -e .
```

---

# Build the Dataset (Milestone 1)

Requires the raw CICIDS2017 CSVs already present at
`data/raw/MachineLearningCVE/` (not tracked in git — see `.gitignore`).

```bash
python scripts/build_dataset.py
```

Produces `data/processed/{train,val,test}.parquet` and
`data/processed/data_quality_report.json`. Takes ~4.5 minutes against the full
2.83M-row dataset. See `docs/DATASET_GUIDE.md` for full pipeline detail.

# Run EDA (Milestone 1)

```bash
python scripts/run_eda.py
```

Reads `data/processed/train.parquet` only (never val/test) and writes
`reports/eda_summary.md`, `reports/eda_feature_stats.csv`, and
`reports/figures/*.png`.

# Run Tests

```bash
pytest -v
```

---

# Run Backend *(not yet implemented — Milestone 3)*

```bash
uvicorn src.api.main:app --reload
```

---

# Run Dashboard *(not yet implemented — Milestone 4)*

```bash
streamlit run src/dashboard/app.py
```

---

# Evaluation Metrics

Primary Metrics

- Macro F1
- Precision
- Recall
- PR-AUC
- ROC-AUC
- MCC

Operational Metrics

- Inference Latency
- Model Size
- Memory Usage

---

# Explainability

SentinelXAI uses TreeSHAP to explain every prediction.

Users can see:

- Why the model predicted an attack
- Which features contributed most
- Global feature importance
- Local feature attribution

---

# Future Work

- Live Network Traffic
- SIEM Integration
- Drift Detection
- Continual Learning
- Cloud Deployment
- Kubernetes
- LLM-powered Incident Reports

---

# License

MIT License

---

# Authors

ML Bubble 2026 Team

SentinelXAI

---

# Acknowledgements

- Canadian Institute for Cybersecurity
- LightGBM
- SHAP
- FastAPI
- Streamlit
- Plotly
- Scikit-learn

---

# Project Status

🚧 Under Active Development

ML Bubble 2026 Competition Project