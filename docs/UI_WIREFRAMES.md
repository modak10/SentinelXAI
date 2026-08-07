# UI_WIREFRAMES.md

# SentinelXAI UI/UX Design Specification

Version: 1.0

Project: SentinelXAI

Competition: ML Bubble 2026

---

# Purpose

This document defines the complete User Interface and User Experience of SentinelXAI.

The dashboard is designed for:

- SOC Analysts
- Cybersecurity Students
- Researchers
- Demonstrations
- Hackathon Presentation

Design Principles

✓ Clean

✓ Professional

✓ Interactive

✓ Explainable

✓ Fast

---

# Technology

Frontend

Streamlit

Visualization

Plotly

Icons

Streamlit Icons

Theme

Dark Cybersecurity Theme

---

# Navigation

```

Dashboard

│

├── Home

├── Live Prediction

├── Explainable AI

├── Decision Intelligence Studio ⭐

├── Failure Explorer ⭐

├── Analytics

└── About

```

---

# Theme

Primary

Blue

Accent

Cyan

Success

Green

Warning

Orange

Danger

Red

Background

Dark Gray

Text

White

---

# Dashboard Layout

```

+--------------------------------------------------------------+

| SentinelXAI |

+--------------------------------------------------------------+

Alerts Critical Accuracy Confidence

---------------------------------------------------------------

Attack Distribution

---------------------------------------------------------------

Severity Distribution

---------------------------------------------------------------

Recent Predictions

---------------------------------------------------------------

Quick Actions

Upload CSV

Predict

Analytics

```

Purpose

Instant overview of the system.

---

# Home Page

Contains

Project Overview

System Health

Model Version

Dataset Information

Latest Predictions

Quick Navigation

---

# Live Prediction

Layout

```

Upload CSV

↓

Preview Dataset

↓

Predict

↓

Results

```

Screen

```

+------------------------------------------------+

Upload Network Traffic

[Choose CSV]

+------------------------------------------------+

[Predict]

--------------------------------------------------

Prediction

Botnet

Confidence

98%

Severity

Critical

--------------------------------------------------

Recommended Actions

✓ Review Firewall

✓ Inspect DNS

✓ Isolate Host

--------------------------------------------------

[Explain Prediction]

```

---

# Explainable AI Page

Purpose

Explain every prediction.

Screen

```

Prediction

Botnet

Confidence

98%

---------------------------------------------------

SHAP Summary Plot

---------------------------------------------------

Top Features

SYN Count

Packet Size

Flow Duration

Destination Port

---------------------------------------------------

Human Explanation

High SYN Count significantly increased
the probability of a Botnet attack.

```

Visualizations

SHAP Summary

Waterfall

Bar Plot

Feature Importance

---

# Decision Intelligence Studio ⭐

Flagship Feature

Purpose

Interactive What-If Analysis

Screen

```

+----------------------------------------------------+

Decision Intelligence Studio

-----------------------------------------------------

Flow Duration

[===========]

Packet Length

[======]

SYN Count

[=============]

ACK Count

[=====]

-----------------------------------------------------

Prediction

PORTSCAN

Confidence

97%

Risk

HIGH

-----------------------------------------------------

Feature Contributions

████████ SYN Count

█████ Packet Size

██ Flow Duration

-----------------------------------------------------

Recommended Investigation

✓ Firewall Logs

✓ IDS History

✓ Check Source IP

```

Judges should immediately understand

Prediction changes

Confidence changes

SHAP changes

Recommendations change

---

# Failure Explorer ⭐

Purpose

Expose model limitations.

Screen

```

Sample ID

1042

-----------------------------------------------

Ground Truth

Bot

Predicted

PortScan

Confidence

55%

-----------------------------------------------

Reason

Feature overlap

Low confidence

Traffic similarity

-----------------------------------------------

SHAP Explanation

████████

████

██

```

Purpose

Build trust.

Show scientific honesty.

---

# Analytics

Purpose

Performance Dashboard

Contains

Accuracy

Precision

Recall

Macro F1

ROC Curve

PR Curve

Confusion Matrix

Training Time

Inference Time

Model Size

Screen

```

+------------------------------------------+

Accuracy

98.7%

Macro F1

98.1%

Inference

29 ms

-------------------------------------------

ROC Curve

-------------------------------------------

PR Curve

-------------------------------------------

Confusion Matrix

-------------------------------------------

Model Comparison

```

---

# About

Contains

Project Description

Architecture

Dataset

Tech Stack

Authors

Competition

Future Work

GitHub Link

---

# Sidebar

Contains

Logo

Navigation

Model Status

Dataset Status

API Status

Theme Switch

---

# User Workflow

```

Open Dashboard

↓

Upload CSV

↓

Predict

↓

View Results

↓

Explain Prediction

↓

Explore Decision Studio

↓

Analyze Failure Cases

↓

View Analytics

```

---

# Color Palette

Background

#0F172A

Primary

#2563EB

Secondary

#0EA5E9

Success

#16A34A

Warning

#F59E0B

Danger

#DC2626

Cards

#1E293B

---

# Typography

Font

Inter

Headings

Bold

Body

Medium

Numbers

SemiBold

---

# Dashboard Components

Cards

Tables

Interactive Charts

Feature Sliders

Upload Widget

Progress Bars

Status Badges

Metric Cards

Expandable Panels

---

# Accessibility

High Contrast

Keyboard Navigation

Responsive Layout

Readable Fonts

Clear Icons

---

# Mobile Support

Future Work

Desktop optimized for hackathon.

---

# UI Principles

Never overload one screen.

Every page answers one question.

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

# Success Criteria

✔ Clean UI

✔ Interactive

✔ Explainable

✔ Fast

✔ Professional

✔ Judge Friendly

✔ Demonstration Ready