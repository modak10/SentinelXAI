"""SentinelXAI Streamlit dashboard (Phase 6).

Dark, security-ops themed multi-page app. Navigation is via the sidebar radio
(the classic, version-robust pattern). Every page degrades gracefully when the
trained model is absent (shows a clear banner instead of crashing).

Pages: Dashboard, Live Prediction, Explain AI, Decision Intelligence Studio,
Failure Explorer, Analytics, About.

UX notes (ui-ux-pro-max review pass):
* Risk is rendered as a labeled colored badge — never color alone (a11y).
* Heavy operations (SHAP, batch CSV) show an explicit spinner.
* CSV uploads are validated (size + required engineered-feature columns) and
  predicted vectorized, not via a per-row Python loop.
* Alerts are ranked by risk, then confidence (``rank_alerts``) on the SOC
  Dashboard.
* The Decision Studio sliders use real engineered-feature statistics instead of
  a hardcoded 0-10000 window.
* Explicit predictions are persisted to SQLite so Dashboard / Failure Explorer /
  Analytics reflect real usage without an extra API call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ``streamlit run src/sentinelxai/dashboard/app.py`` executes this file as a
# top-level script (no parent package), so the standard relative import would
# fail. Add the package root to sys.path, then use an absolute import so the
# app launches from the repo root, from AppTest, and from the Docker image
# (where PYTHONPATH=/app/src is already set).
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sentinelxai.decision import (  # noqa: E402 - import after sys.path setup
    build_decision_payload,
    rank_alerts,
)
from sentinelxai.explainability import humanize_contributions  # noqa: E402

from sentinelxai.dashboard.service import get_service  # noqa: E402

st.set_page_config(
    page_title="SentinelXAI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- theme -----------------------------------------------------------------
# Risk badge colors (green->red severity ladder) used alongside the risk label
# so severity is never communicated by color alone.
RISK_COLORS = {
    "LOW": "#22C55E",
    "MEDIUM": "#F1C40F",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — mirrors the API upload policy.

EDITABLE_FEATURES = [
    "SYN Flag Count",
    "Flow Duration",
    "Destination Port",
    "Packet Length Mean",
    "Bwd Packet Length Mean",
]


def _risk_badge(risk: str) -> str:
    """Return a rounded HTML badge pairing the risk label with its color."""
    label = str(risk).upper()
    color = RISK_COLORS.get(label, "#64748B")
    return (
        '<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f"background:{color};color:#020617;font-weight:600;font-size:0.85rem;"
        f'min-width:72px;text-align:center;">{label}</span>'
    )


def _feature_stats(svc) -> dict[str, tuple[float, float, float]]:
    """(min, max, median) per engineered feature from the EDA report.

    Lets the Studio sliders use realistic ranges rather than a flat 0-10000
    window (Flow Duration spans microseconds->minutes; counts differ wildly).
    """
    stats_path: Path = svc["cfg"].paths.reports_dir / "eda_feature_stats.csv"
    if not stats_path.exists():
        return {}
    df = pd.read_csv(stats_path).set_index("Unnamed: 0")
    out: dict[str, tuple[float, float, float]] = {}
    for feat, row in df.iterrows():
        lo = float(row.get("min", 0))
        hi = float(row.get("max", 10000))
        if hi <= lo:
            continue
        mid = float(row.get("50%", lo))
        mid = max(lo, min(mid, hi))
        out[feat] = (lo, hi, mid)
    return out


def _banner(svc) -> bool:
    if not svc["available"]:
        st.warning(
            "⚠️ Trained model not found. Run `python scripts/train_final_lightgbm.py` "
            "(after `make data`) to enable predictions. Pages show static info only."
        )
        return False
    return True


def _predict(svc, record: dict, log: bool = False):
    """Single-record prediction + explanation + decision, in-process.

    ``log=True`` persists to history — used on explicit user actions (Predict,
    batch upload) but NOT on every Studio slider tick.
    """
    result = svc["inference"].predict_single(record)
    pc = result.predicted_classes[0]
    conf = result.confidences[0]
    proba = result.probabilities[0]
    explanation = None
    human: list[str] = []
    contribs = []
    if svc["explainer"] is not None:
        from sentinelxai.explainability.shap_engine import LocalContribution

        local = svc["explainer"].explain_single(record, pc)
        explanation = local
        human = humanize_contributions(local)
        contribs = [
            LocalContribution(
                name=c.name, value=c.value, direction=c.direction, rank=c.rank, weight=c.weight
            )
            for c in local.top_contributions
        ]
    decision = build_decision_payload(
        predicted_class=pc,
        confidence=conf,
        top_contributions=contribs,
        cfg=svc["decision_cfg"],
        human_explanation=human,
    )
    if log:
        try:
            svc["store"].log_prediction(
                predicted_class=pc,
                confidence=conf,
                risk=decision["risk"],
                input_features=record,
                probabilities=proba,
            )
        except Exception:  # noqa: BLE001 - history must never break a prediction
            pass
    return pc, conf, proba, explanation, decision


def _shap_frame(expl) -> pd.Series:
    return pd.DataFrame(
        [{"feature": c.name, "SHAP": c.value} for c in expl.top_contributions]
    ).set_index("feature")["SHAP"]


def _proba_frame(proba: dict[str, float], top: int = 5) -> pd.Series:
    rows = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return pd.Series({label: val for label, val in rows})


# --- pages -----------------------------------------------------------------


def page_dashboard(svc):
    """What is happening right now? Ranked alert queue + distribution."""
    st.title("🛡 SentinelXAI — SOC Dashboard")
    hist = svc["store"].get_history(limit=500)
    if not hist:
        st.info(
            "No predictions logged yet. Use **Live Prediction** (or upload a CSV) "
            "to generate alerts — they will be ranked here by risk and confidence."
        )
        return

    df = pd.DataFrame([h.to_dict() for h in hist])
    ranked = rank_alerts(
        [
            {
                "id": h.id,
                "timestamp": h.timestamp,
                "predicted_class": h.predicted_class,
                "confidence": h.confidence,
                "risk": h.risk,
            }
            for h in hist
        ],
        svc["decision_cfg"],
    )
    ranked_df = pd.DataFrame(ranked).sort_values("priority")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", len(df))
    c2.metric("Critical", int((df["risk"] == "CRITICAL").sum()))
    c3.metric("High", int((df["risk"] == "HIGH").sum()))
    c4.metric("Model Status", "ONLINE" if svc["available"] else "OFFLINE")

    st.subheader("Priority Queue (risk → confidence)")
    st.dataframe(
        ranked_df[["priority", "timestamp", "predicted_class", "confidence", "risk"]],
        width="stretch",
    )

    st.subheader("Attack Distribution (logged alerts)")
    if not df["predicted_class"].isna().all():
        st.bar_chart(df["predicted_class"].value_counts())

    st.subheader("Recent Alerts")
    recent = df[["timestamp", "predicted_class", "confidence", "risk"]].copy()
    recent["confidence"] = (recent["confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(recent, width="stretch")


def _render_prediction(pc, conf, dec, expl_ex):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Prediction: {pc}")
        st.markdown(f"**Risk:** {_risk_badge(dec['risk'])}", unsafe_allow_html=True)
        st.metric("Confidence", f"{conf:.1%}")
        st.caption(f"Confidence band: {dec['confidence_band']}")
        if dec["is_failure_candidate"]:
            st.error("⚠ Low-confidence prediction — treat with caution.")
        st.subheader("Recommended Investigation")
        for r in dec["recommendations"]:
            st.markdown(f"- {r}")
    with col2:
        st.subheader("Why? (SHAP)")
        if expl_ex is not None:
            for sent in expl_ex.human_explanation or humanize_contributions(expl_ex):
                st.markdown(f"- {sent}")
            st.bar_chart(_shap_frame(expl_ex))
        else:
            st.info("SHAP explanations require the trained model.")


def page_predict(svc):
    st.title("🔍 Live Prediction")
    if not _banner(svc):
        return
    mode = st.radio("Input mode", ["JSON record", "CSV upload"], horizontal=True)

    if mode == "JSON record":
        sample = '{\n  "Flow Duration": 0.0,\n  "SYN Flag Count": 1.0\n}'
        text = st.text_area("Engineered feature record (JSON)", value=sample, height=200)
        if st.button("Predict", type="primary"):
            try:
                record = json.loads(text)
                with st.spinner("Running prediction + SHAP explanation…"):
                    pc, conf, proba, expl_ex, dec = _predict(svc, record, log=True)
                _render_prediction(pc, conf, dec, expl_ex)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid input: {exc}")
        return

    # --- CSV batch mode ---
    file = st.file_uploader("Upload CSV of engineered features", type="csv")
    if file is None:
        return
    if st.button("Predict batch", type="primary"):
        if getattr(file, "size", 0) and file.size > MAX_UPLOAD_BYTES:
            st.error(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
            return
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse CSV: {exc}")
            return
        required = svc["inference"].feature_names
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"CSV is missing required engineered features: {missing[:10]}")
            return
        records = df[required].astype(float).to_dict(orient="records")
        with st.spinner(f"Predicting {len(records)} rows…"):
            result = svc["inference"].predict(records)
            out = []
            for idx, (rec, pc, conf) in enumerate(
                zip(records, result.predicted_classes, result.confidences, strict=True)
            ):
                dec = build_decision_payload(
                    predicted_class=pc,
                    confidence=conf,
                    top_contributions=[],  # no per-row SHAP in batch (fast path)
                    cfg=svc["decision_cfg"],
                )
                out.append(
                    {
                        "predicted_class": pc,
                        "confidence": round(float(conf), 4),
                        "risk": dec["risk"],
                        "risk_rank": dec["risk_rank"],
                    }
                )
                try:
                    svc["store"].log_prediction(
                        predicted_class=pc,
                        confidence=float(conf),
                        risk=dec["risk"],
                        input_features=rec,
                        probabilities=result.probabilities[idx],
                    )
                except Exception:  # noqa: BLE001 - best effort
                    pass
        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(pd.DataFrame(out), width="stretch")


def page_explain(svc):
    st.title("🧠 Explain AI")
    if not _banner(svc):
        return
    st.markdown(
        "Paste an engineered-feature record to see the local **TreeSHAP** "
        "explanation and human-readable reasoning behind the prediction."
    )
    text = st.text_area("Engineered feature record (JSON)", height=150)
    if st.button("Explain", type="primary"):
        try:
            record = json.loads(text)
            with st.spinner("Computing SHAP contributions…"):
                pc, conf, proba, expl_ex, dec = _predict(svc, record, log=False)
            _render_prediction(pc, conf, dec, expl_ex)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid input: {exc}")


def page_studio(svc):
    st.title("🎬 Decision Intelligence Studio")
    if not _banner(svc):
        return
    feats = svc["inference"].feature_names
    editable = [f for f in EDITABLE_FEATURES if f in feats]
    st.caption(
        "Drag the sliders and watch the prediction, confidence, and risk update "
        "live. Ranges and defaults come from the engineered-feature statistics, "
        "so what-if analysis matches the data. SHAP is recomputed on every change."
    )
    stats = _feature_stats(svc)
    record: dict[str, float] = {}
    cols = st.columns(max(len(editable), 1))
    for i, f in enumerate(editable):
        lo, hi, mid = stats.get(f, (0.0, 10000.0, 0.0))
        record[f] = cols[i].slider(f, float(lo), float(hi), float(mid), key=f)

    full = {f: record.get(f, 0.0) for f in feats}
    with st.spinner("Updating prediction + SHAP…"):
        pc, conf, proba, expl_ex, dec = _predict(svc, full)

    st.subheader(f"Live Prediction: {pc}")
    st.markdown(f"**Risk:** {_risk_badge(dec['risk'])}", unsafe_allow_html=True)
    st.metric("Confidence", f"{conf:.1%}")
    if expl_ex is not None:
        st.bar_chart(_shap_frame(expl_ex))


def _render_failure(svc, failure) -> None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Model predicted:** `{failure.predicted_class}`")
        st.markdown(f"**Risk:** {_risk_badge(failure.risk)}", unsafe_allow_html=True)
        st.progress(min(float(failure.confidence), 1.0))
        thr = svc["decision_cfg"].modulation.failure_confidence_threshold
        st.caption(f"Confidence {failure.confidence:.1%} — below the {thr:.0%} threshold")
        st.subheader("Recommended Investigation")
        for r in build_decision_payload(
            predicted_class=failure.predicted_class,
            confidence=failure.confidence,
            top_contributions=[],
            cfg=svc["decision_cfg"],
        )["recommendations"]:
            st.markdown(f"- {r}")
    with col2:
        if failure.probabilities:
            st.subheader("Top class probabilities")
            st.bar_chart(_proba_frame(failure.probabilities))
        st.subheader("Why does the model doubt itself?")
        if svc["explainer"] is not None and failure.input_features:
            try:
                with st.spinner("Recomputing SHAP for the logged record…"):
                    local = svc["explainer"].explain_single(
                        failure.input_features, failure.predicted_class
                    )
                for sent in local.human_explanation or humanize_contributions(local):
                    st.markdown(f"- {sent}")
                st.bar_chart(_shap_frame(local))
            except Exception:  # noqa: BLE001 - stale artifact / changed schema
                st.markdown(
                    "- The stored record cannot be re-explained (model or feature "
                    "set changed after it was logged)."
                )
        else:
            st.markdown(
                "- No stored feature record is available (or the model is not "
                "loaded), so only the logged class probabilities are shown."
            )


def page_failure(svc):
    st.title("🩺 Failure Explorer")
    st.caption(
        "Surfaces low-confidence predictions — failure candidates — so an analyst "
        "can judge *why* the model is least sure, plus the top competing classes."
    )
    hist = svc["store"].get_history(limit=1000)
    failures = [
        h for h in hist if (h.confidence or 1.0) < svc["decision_cfg"].modulation.failure_confidence_threshold
    ]
    if not failures:
        st.info(
            "No low-confidence predictions logged yet. Predictions below the "
            "failure threshold will appear here automatically."
        )
        return
    sel = st.selectbox(
        "Select a failure candidate",
        [f"#{h.id} · {h.predicted_class} · {h.confidence:.1%}" for h in failures],
    )
    failure_id = int(sel.split("·")[0].strip().lstrip("#"))
    failure = next(h for h in failures if h.id == failure_id)
    _render_failure(svc, failure)


def page_analytics(svc):
    st.title("📊 Model Analytics")
    metrics_path = svc["cfg"].paths.reports_dir / "lightgbm" / "lightgbm_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
        c2.metric("Macro F1", f"{m.get('f1_macro', 0):.4f}")
        c3.metric("Weighted F1", f"{m.get('f1_weighted', 0):.4f}")
        c4.metric("Infer (ms/sample)", f"{m.get('inference_time_per_sample_ms', 0):.4f}")
    else:
        st.info("Train the final model to populate metrics.")
    cm = svc["cfg"].paths.reports_dir / "lightgbm" / "figures" / "lightgbm_confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption="Confusion Matrix (validation)", width="stretch")


def page_about(svc):
    st.title("ℹ️ About / Architecture")
    st.markdown(
        """
        **SentinelXAI** — Explainable AI Decision Intelligence Platform for Network
        Intrusion Detection.

        Pipeline: `Network Flow → Validation → LightGBM → TreeSHAP → Decision →
        Analyst Dashboard`.

        - Model: LightGBM (Macro F1 0.9201 on validation)
        - Explainability: TreeSHAP
        - Decision layer: confidence, risk policy, recommendations, alert ranking
        - Backend: FastAPI · Frontend: Streamlit · DB: SQLite · Deploy: Docker
        """
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Live Prediction": page_predict,
    "Explain AI": page_explain,
    "Decision Intelligence Studio": page_studio,
    "Failure Explorer": page_failure,
    "Analytics": page_analytics,
    "About": page_about,
}


def main():
    svc = get_service()
    st.sidebar.title("SentinelXAI")
    st.sidebar.caption("Explainable IDS Decision Support")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[choice](svc)


if __name__ == "__main__":
    main()