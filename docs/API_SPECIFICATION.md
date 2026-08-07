# API_SPECIFICATION.md

# SentinelXAI REST API Specification

Version: 1.0

Project: SentinelXAI – Explainable AI Decision Intelligence Platform

Competition: ML Bubble 2026

---

# Purpose

This document defines the REST API exposed by SentinelXAI.

The API is responsible for:

- Receiving prediction requests
- Running ML inference
- Generating explainability
- Returning recommendations
- Providing system metrics
- Exposing model metadata

The API follows REST principles.

---

# Base URL

Development

http://localhost:8000

Swagger

http://localhost:8000/docs

ReDoc

http://localhost:8000/redoc

---

# Technology

Framework

FastAPI

Serialization

JSON

Validation

Pydantic

Documentation

OpenAPI

---

# API Architecture

Client

↓

FastAPI

↓

Validation

↓

Preprocessing

↓

LightGBM

↓

SHAP

↓

Decision Engine

↓

JSON Response

---

# Response Format

Every response follows

{
    "success": true,
    "message": "",
    "data": {}
}

Errors

{
    "success": false,
    "message": "Invalid CSV format",
    "errors": []
}

---

# Endpoint Summary

| Method | Endpoint | Purpose |
|----------|---------------------|------------------------------|
| POST | /predict | Predict one sample |
| POST | /batch_predict | Predict multiple samples |
| POST | /upload | Upload CSV |
| GET | /health | Health Check |
| GET | /metrics | Model Metrics |
| GET | /model | Model Information |
| GET | /feature-importance | Global Feature Importance |
| GET | /history | Prediction History |

---

# POST /predict

Purpose

Predict one network flow.

Request

{
    "features": {
        "Flow Duration": 125,
        "Total Fwd Packets": 10,
        "Packet Length Mean": 534.2
    }
}

Response

{
    "success": true,
    "prediction": "PortScan",
    "confidence": 0.98,
    "severity": "High",
    "recommendation": [
        "Inspect firewall logs",
        "Review source IP"
    ]
}

Status Codes

200

Prediction successful

400

Invalid request

500

Internal server error

---

# POST /batch_predict

Purpose

Predict multiple rows.

Request

CSV Upload

or

JSON Array

Response

{
    "total_records": 250,
    "predictions": [...]
}

---

# POST /upload

Purpose

Upload dataset.

Accept

CSV

Validation

Maximum Size

50 MB

Allowed Types

text/csv

Validation

Column names

Feature count

Missing values

---

# GET /health

Purpose

Health check.

Response

{
    "status":"healthy",
    "model_loaded":true,
    "database":"connected"
}

---

# GET /metrics

Purpose

Return evaluation metrics.

Example

{
    "accuracy":0.99,
    "macro_f1":0.98,
    "precision":0.98,
    "recall":0.97,
    "roc_auc":0.99,
    "inference_latency_ms":28
}

---

# GET /model

Purpose

Return metadata.

Response

{
    "name":"LightGBM",
    "version":"1.0",
    "dataset":"CICIDS2017",
    "training_date":"2026-08-01"
}

---

# GET /feature-importance

Purpose

Return global feature importance.

Response

[
{
"name":"Flow Duration",
"importance":0.45
},
{
"name":"Packet Length Mean",
"importance":0.31
}
]

---

# GET /history

Purpose

Prediction history.

Response

[
{
"id":1,
"prediction":"Bot",
"confidence":0.98,
"timestamp":"..."
}
]

---

# Pydantic Models

PredictionRequest

PredictionResponse

BatchPredictionRequest

BatchPredictionResponse

HealthResponse

MetricsResponse

ModelResponse

HistoryResponse

---

# Validation Rules

Validate

CSV Schema

Feature Count

Missing Values

Invalid Numbers

Unexpected Columns

Empty Files

Reject invalid input.

---

# Error Codes

400

Bad Request

401

Unauthorized (Future)

404

Not Found

413

File Too Large

422

Validation Error

500

Internal Server Error

---

# Logging

Every request logs

Timestamp

Endpoint

Latency

Status

IP (optional)

Prediction Count

Errors

---

# Security

Future

Authentication

JWT

Rate Limiting

HTTPS

Role-based Access

Audit Logs

---

# Performance Targets

Single Prediction

<100 ms

Batch Prediction

<5 seconds

API Availability

99%

---

# OpenAPI

Swagger UI

/docs

Automatically generated.

No manual documentation required.

---

# Future Endpoints

POST /feedback

POST /retrain

GET /drift

GET /alerts

GET /recommendations

GET /statistics

---

# Example Prediction Flow

Client

↓

POST /predict

↓

Validate JSON

↓

Preprocess Features

↓

LightGBM

↓

Probability

↓

TreeSHAP

↓

Risk Engine

↓

Recommendation Engine

↓

Return JSON

---

# Acceptance Criteria

✔ Swagger available

✔ OpenAPI generated

✔ Input validated

✔ Errors handled

✔ Responses documented

✔ JSON consistent

✔ Logging enabled

✔ API tested

---

# API Design Principles

- RESTful
- Stateless
- JSON only
- Strong validation
- Versionable
- Fully documented
- Human-readable responses
- Consistent error handling