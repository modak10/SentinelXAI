# EDA Summary (train split)

- Rows: 1,764,559
- Numeric feature columns: 77
- Classes: 15

## Class Distribution

| Class | Count | % of train |
|---|---|---|
| BENIGN | 1,466,540 | 83.1109% |
| DoS Hulk | 120,992 | 6.8568% |
| DDoS | 89,610 | 5.0783% |
| PortScan | 63,486 | 3.5978% |
| DoS GoldenEye | 7,200 | 0.4080% |
| FTP-Patator | 4,152 | 0.2353% |
| DoS slowloris | 3,769 | 0.2136% |
| DoS Slowhttptest | 3,660 | 0.2074% |
| SSH-Patator | 2,253 | 0.1277% |
| Bot | 1,364 | 0.0773% |
| Web Attack - Brute Force | 1,029 | 0.0583% |
| Web Attack - XSS | 456 | 0.0258% |
| Infiltration | 25 | 0.0014% |
| Web Attack - Sql Injection | 15 | 0.0009% |
| Heartbleed | 8 | 0.0005% |

## Zero-Variance Features

These features are constant in the train split and carry no signal — candidates to drop in Milestone 2:

- Bwd PSH Flags
- Bwd URG Flags
- Fwd Avg Bytes/Bulk
- Fwd Avg Packets/Bulk
- Fwd Avg Bulk Rate
- Bwd Avg Bytes/Bulk
- Bwd Avg Packets/Bulk
- Bwd Avg Bulk Rate

## Highly Correlated Feature Pairs (|r| >= 0.95)

| Feature A | Feature B | r |
|---|---|---|
| Total Fwd Packets | Subflow Fwd Packets | 1.0 |
| Total Backward Packets | Subflow Bwd Packets | 1.0 |
| Total Length of Fwd Packets | Subflow Fwd Bytes | 1.0 |
| Total Length of Bwd Packets | Subflow Bwd Bytes | 1.0 |
| Fwd Packet Length Mean | Avg Fwd Segment Size | 1.0 |
| Bwd Packet Length Mean | Avg Bwd Segment Size | 1.0 |
| Fwd PSH Flags | SYN Flag Count | 1.0 |
| Fwd URG Flags | CWE Flag Count | 1.0 |
| RST Flag Count | ECE Flag Count | 1.0 |
| Total Fwd Packets | Total Backward Packets | 0.9992 |
| Total Fwd Packets | Subflow Bwd Packets | 0.9992 |
| Total Backward Packets | Subflow Fwd Packets | 0.9992 |
| Subflow Fwd Packets | Subflow Bwd Packets | 0.9992 |
| Flow Duration | Fwd IAT Total | 0.9986 |
| Flow IAT Max | Fwd IAT Max | 0.9981 |
| Packet Length Mean | Average Packet Size | 0.9978 |
| Total Fwd Packets | Total Length of Bwd Packets | 0.997 |
| Total Fwd Packets | Subflow Bwd Bytes | 0.997 |
| Total Length of Bwd Packets | Subflow Fwd Packets | 0.997 |
| Subflow Fwd Packets | Subflow Bwd Bytes | 0.997 |
| Total Backward Packets | Total Length of Bwd Packets | 0.9951 |
| Total Backward Packets | Subflow Bwd Bytes | 0.9951 |
| Total Length of Bwd Packets | Subflow Bwd Packets | 0.9951 |
| Subflow Bwd Packets | Subflow Bwd Bytes | 0.9951 |
| Idle Mean | Idle Max | 0.9903 |
| Idle Mean | Idle Min | 0.9901 |
| Flow IAT Max | Idle Max | 0.9894 |
| Fwd IAT Max | Idle Max | 0.9886 |
| Max Packet Length | Packet Length Std | 0.984 |
| Bwd Packet Length Max | Bwd Packet Length Std | 0.9825 |
| Flow Packets/s | Fwd Packets/s | 0.9805 |
| Flow IAT Max | Idle Mean | 0.9798 |
| Fwd IAT Max | Idle Mean | 0.9783 |
| Fwd Packet Length Max | Fwd Packet Length Std | 0.9684 |
| Idle Max | Idle Min | 0.9615 |
| Bwd Packet Length Max | Bwd Packet Length Mean | 0.9581 |
| Bwd Packet Length Max | Avg Bwd Segment Size | 0.9581 |
| Flow IAT Max | Idle Min | 0.9514 |

## Notes

- Computed on the TRAIN split only, per this project's own leakage-avoidance policy (docs/JUDGE_QNA.md Q8) — val/test were not read.
- Severe class imbalance confirmed (see docs/DATASET_GUIDE.md and data/processed/data_quality_report.json) — informs the Macro-F1 metric choice for Milestone 2.