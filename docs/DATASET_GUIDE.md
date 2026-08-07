# DATASET_GUIDE.md

# SentinelXAI Dataset Guide

Version: 2.0 — updated after Milestone 1 implementation and a real run against the full raw dataset.
Supersedes the v1.0 placeholder. Deviations from v1.0 are called out explicitly below
rather than silently applied — see "Deviations From the Original Plan".

Project: SentinelXAI
Competition: ML Bubble 2026

---

# Purpose

This document describes the actual, implemented, and verified data engineering
workflow for SentinelXAI — not an aspirational one. Every number below was produced
by running `python scripts/build_dataset.py` against the real raw dataset.

---

# Selected Dataset

Dataset Name: **CICIDS2017**
Official Source: Canadian Institute for Cybersecurity (CIC)
Format: MachineLearningCSV variant (flow-level features, not raw PCAP)
File Type: CSV

---

# Dataset Location — Path Correction

Both `AGENT.md` and the original `DATASET_GUIDE.md` reference
`data/raw/MachineLearningCSV/`. The folder actually present on disk is:

```
data/raw/MachineLearningCVE/
```

This is treated as a naming typo in the documentation, not a data problem — the code
(`configs/data.yaml` -> `src/sentinelxai/data/loader.py`) points at the real path.
Flagged here so it isn't silently "corrected" without a record of the discrepancy.

Never modify files inside `raw/`.

---

# Raw File Inventory (verified)

| File | Rows (excl. header) | Label(s) present |
|---|---|---|
| Monday-WorkingHours.pcap_ISCX.csv | 529,918 | BENIGN |
| Tuesday-WorkingHours.pcap_ISCX.csv | 445,909 | BENIGN, FTP-Patator, SSH-Patator |
| Wednesday-workingHours.pcap_ISCX.csv | 692,703 | BENIGN, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, Heartbleed |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 170,366 | BENIGN, Web Attack - Brute Force, Web Attack - Sql Injection, Web Attack - XSS |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 288,602 | BENIGN, Infiltration |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 191,033 | BENIGN, Bot |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 286,467 | BENIGN, PortScan |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 225,745 | BENIGN, DDoS |
| **Total** | **2,830,743** | **15 distinct classes** |

Every raw file shares an identical 79-column header (verified via checksum) — no
per-file column-name normalization is needed beyond whitespace stripping.

---

# Known Data Quality Issues (found during implementation, not assumed)

1. **Duplicate `Fwd Header Length` column.** The header repeats this column name at
   positions 35 and 56 with identical values in every row. `schema.resolve_duplicate_columns`
   handles this **generically** — it detects pandas' own `.1`/`.2` mangled-duplicate
   suffixes (not a hardcoded name match), so it catches any number of duplicates of any
   column on any future dataset variant without a config entry per column. (Milestone 2
   review found the first version of this function only matched exact pre-mangle names
   and therefore never actually fired — see the "Milestone 2 Corrections" section below.)
   `configs/data.yaml` -> `duplicate_columns` is now only used to *override* the default
   keep-first policy for a specific column, and is empty by default.
2. **Corrupted "Web Attack" labels.** The three Web Attack labels contain the Unicode
   replacement character (U+FFFD) in place of what was originally an en-dash — this is
   baked into the publicly released CSVs (valid UTF-8, not a read/encoding error on our
   side). Normalized by `schema.normalize_label` to e.g. `"Web Attack - Brute Force"`.
3. **±Infinity values** in `Flow Bytes/s` and `Flow Packets/s`, caused by division by a
   zero `Flow Duration`. Replaced with NaN, then the affected rows are dropped (see
   Cleaning Rules below).
4. **A very large number of exact duplicate rows**: 307,078 out of 2,830,743 (10.85%).
   This matches independent published re-audits of this exact CSV release — it is a
   known characteristic of the dataset, not a bug in our pipeline.
5. **No Flow ID / Source IP / Destination IP / Timestamp columns** in this
   MachineLearningCVE variant (unlike the full PCAP-derived CICIDS2017 release) — the
   generic "identifier-column leakage" risk flagged in the initial architecture audit
   does **not** apply to this specific file variant. Verified by inspecting the raw
   header; `configs/data.yaml` -> `leak_risk_columns` is empty for this reason, not by
   oversight.

---

# Dataset Statistics (after cleaning, before split)

| Metric | Value |
|---|---|
| Rows before cleaning | 2,830,743 |
| ±Infinity values replaced | 4,376 |
| Rows dropped (NaN, post-infinity-replacement) | 2,867 |
| Rows dropped (exact duplicates) | 307,078 |
| Label values normalized (encoding artifact cleanup) | 2,143 |
| **Rows after cleaning** | **2,520,798** |
| Feature columns | 77 (+ `Label`, + `__source_file` metadata) |
| Classes | 15 (1 benign + 14 attack) |

Full machine-readable detail — including the exact per-class counts — is regenerated
on every pipeline run at `data/processed/data_quality_report.json`.

## Milestone 2 Corrections

- **Duplicate-column fix applied and the dataset regenerated.** The feature count above
  changed from the original 78 to 77 as a direct result: `Fwd Header Length.1` (the
  undetected duplicate, see issue 1 above) is now correctly dropped at the source during
  `build_dataset.py`, rather than surviving into the processed parquet files and only
  being caught downstream by EDA's correlation scan. All row/duplicate/cleaning counts
  in the table above are unchanged (the fix only removes a redundant *column*, not any
  rows).

---

# Target Labels (verified, exact strings)

```
BENIGN
DDoS
PortScan
Bot
Infiltration
FTP-Patator
SSH-Patator
DoS GoldenEye
DoS Hulk
DoS Slowhttptest
DoS slowloris
Heartbleed
Web Attack - Brute Force
Web Attack - Sql Injection
Web Attack - XSS
```

Canonical taxonomy lives in code at `src/sentinelxai/data/schema.py::KNOWN_LABELS`
(never hardcode this list elsewhere). A coarse `ATTACK_FAMILY` grouping is also
defined there for EDA/dashboard summaries only — the fine-grained label above remains
the actual ML target.

---

# Class Imbalance (measured, not illustrative)

| Class | Row Count | % of Total |
|---|---|---|
| BENIGN | 2,095,057 | 83.11% |
| DoS Hulk | 172,846 | 6.86% |
| DDoS | 128,014 | 5.08% |
| PortScan | 90,694 | 3.60% |
| DoS GoldenEye | 10,286 | 0.41% |
| FTP-Patator | 5,931 | 0.24% |
| DoS slowloris | 5,385 | 0.21% |
| DoS Slowhttptest | 5,228 | 0.21% |
| SSH-Patator | 3,219 | 0.13% |
| Bot | 1,948 | 0.08% |
| Web Attack - Brute Force | 1,470 | 0.06% |
| Web Attack - XSS | 652 | 0.03% |
| Infiltration | 36 | 0.0014% |
| Web Attack - Sql Injection | 21 | 0.0008% |
| Heartbleed | 11 | 0.0004% |

This is severe: the rarest class (Heartbleed) is ~190,000x smaller than the majority
class. Confirms the Macro F1 metric choice for Milestone 2 (docs/JUDGE_QNA.md Q12) —
accuracy alone would be trivially dominated by BENIGN.

---

# Dataset Pipeline (as implemented)

```
Raw CSVs (8 files)
    │
    ▼
Merge (loader.py) — strip column names, resolve duplicate columns
    │
    ▼
Clean (cleaning.py) — ±Infinity -> NaN -> drop, drop duplicate rows, normalize labels
    │
    ▼
Validate schema (validation.py) — label column present, columns unique, non-empty
    │
    ▼
Validate labels (validation.py) — class counts, unknown-label detection, rare-class detection
    │
    ▼
Stratified split (split.py) — 70/15/15, documented rare-class fallback (see below)
    │
    ▼
Save (build_dataset.py) — train/val/test parquet + data_quality_report.json
```

Runs end-to-end with one command in ~4.5 minutes against the full 2.83M-row raw
dataset:

```bash
python scripts/build_dataset.py
```

---

# Cleaning Rules (as implemented, in order)

1. Replace ±Infinity with NaN in `Flow Bytes/s` and `Flow Packets/s`
   (`configs/data.yaml` -> `cleaning.infinity_columns`).
2. Drop any row containing a NaN in *any* column (deliberately global, not scoped only
   to the infinity columns, as a safety net against unrelated parsing issues) —
   `configs/data.yaml` -> `cleaning.nan_strategy: drop_row`.
3. Drop exact duplicate rows, judged on all feature+label columns (excluding the
   internal `__source_file` trace column so a flow repeated across files is still
   caught).
4. Normalize label strings (whitespace, the U+FFFD Web Attack corruption).

Every step's row-count impact is recorded in `data_quality_report.json` — nothing is
silently discarded.

---

# Data Validation Checklist (implemented)

- [x] Duplicate rows — detected and dropped, count reported
- [x] Missing values — detected (post-infinity-replacement) and dropped, count reported
- [x] Infinite values — detected and replaced, count reported
- [x] Invalid/unknown labels — detected and reported (would not silently pass)
- [x] Duplicate columns — detected structurally (`validate_schema` raises if any survive)
- [x] Constant (zero-variance) columns — detected in EDA (`eda.py::zero_variance_features`),
      **not removed** at the data-engineering stage; left as a documented finding for
      Milestone 2 feature engineering to act on
- [ ] Incorrect data types — not yet explicitly type-checked column-by-column; all raw
      columns except `Label` parse as numeric via pandas' default inference, unverified
      further. Flagged as a Milestone 2 follow-up if type coercion bugs surface.

---

# Rare-Class Split Strategy (Project Lead decision — implemented)

Explicit Project Lead decision: **rare classes are never dropped or excluded.**
`src/sentinelxai/data/split.py` implements a custom stratified splitter (not
`sklearn.model_selection.train_test_split`, which raises rather than degrading
gracefully for classes smaller than the number of splits) with this fallback ladder,
per class of size `n`:

- `n < 3` -> all rows to train; no val/test representation is possible and this is
  reported, not hidden.
- Proportional 70/15/15 would give 0 rows to val or test -> reserve exactly 1 row for
  val and 1 for test, remainder to train.
- Otherwise -> standard proportional split (rounded, remainder assigned to train so
  counts always sum exactly to `n`).

**Verified result on the real dataset:** only one class (Heartbleed, n=11) fell below
`rare_class_floor` (20). It received train=8 / val=2 / test=1 — every split has at
least one Heartbleed sample; no fallback below the "n < 3" tier was needed on the real
data (the tier exists and is tested regardless, since the real data could change).
Full detail: `data_quality_report.json` -> `split.fallback_classes` and
`split.per_class_allocation`.

This does **not** solve the small-sample statistical-reliability problem (a Heartbleed
test recall computed on 1 sample is not meaningful) — it solves the *data-integrity*
problem of silently discarding a real attack class. The reliability caveat must be
stated explicitly wherever Heartbleed metrics are reported in Milestone 2.

---

# Output Files (as implemented)

```
data/processed/
    train.parquet                  1,764,559 rows
    val.parquet                      378,120 rows
    test.parquet                     378,119 rows
    data_quality_report.json       full cleaning + label + split report
reports/
    eda_summary.md                  human-readable EDA findings (train split only)
    eda_feature_stats.csv           describe() + skew per feature (train split only)
    figures/
        class_distribution.png
        correlation_heatmap.png
```

Parquet was chosen over CSV (the original plan) for the processed splits: ~4-5x
smaller on disk, preserves dtypes exactly (no re-parsing ambiguity on load), and reads
substantially faster — all of which matter for Milestone 2 iterating on model training
against these files repeatedly.

No intermediate `cicids2017_clean.csv` is materialized — the cleaned-but-unsplit
DataFrame exists only in memory between pipeline stages, since nothing downstream
needs it as a standalone artifact.

---

# Deviations From the Original Plan (v1.0 of this document)

Recorded explicitly per AGENT.md's rule: never silently contradict prior documented
decisions.

| v1.0 said | What was actually built | Why |
|---|---|---|
| 5 standalone scripts (`download_dataset.py`, `merge_dataset.py`, `clean_dataset.py`, `split_dataset.py`, `validate_dataset.py`) | 1 CLI (`scripts/build_dataset.py`) orchestrating a tested library (`src/sentinelxai/data/*.py`) | 5 standalone scripts would each need their own config/logging boilerplate, or awkwardly import from each other. The library modules already have one responsibility each (loader/cleaning/validation/split); the script is a thin orchestrator, which is more testable than 5 separate `__main__` entrypoints. |
| CSV outputs (`train.csv`, `validation.csv`, `test.csv`) | Parquet outputs | Smaller, dtype-safe, faster — see Output Files above. |
| `configs/label_mapping.json` created during data engineering | Label encoding deferred to Milestone 2 | The data layer's consumers (EDA, dashboard, SHAP output) all benefit from string labels; encoding is a training-time concern and belongs with the model that needs it. |
| `logs/preprocessing.log` | Routed through the existing `application.log` / `error.log` (configs/logging.yaml) | Avoids a second logging config path for one pipeline stage; can be split out later if log volume warrants it. |
| `data/raw/MachineLearningCSV/` | `data/raw/MachineLearningCVE/` | Matches what is actually on disk — see "Dataset Location" above. |

These are presented for Project Lead review, not unilaterally finalized — see the
Milestone 1 summary for the explicit approval ask.

---

# Engineering Principle

Data quality determines model quality. Every number in this document was produced by
an actual run against the real dataset, not estimated — reproduce it at any time with
`python scripts/build_dataset.py` followed by `python scripts/run_eda.py`.
