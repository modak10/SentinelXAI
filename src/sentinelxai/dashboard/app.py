"""SentinelXAI Streamlit dashboard (Phase 6).

Dark, security-ops themed multi-page app. Navigation is via the sidebar radio
(the classic, version-robust pattern). Every page degrades gracefully when the
trained model is absent (shows a clear banner instead of crashing).

Pages: Dashboard, Live Prediction, Explainable AI, Decision Intelligence Studio,
Failure Explorer, Analytics, About.
"""

from __future__ import annotations


import pandas as pd
import streamlit as st

from sentinelxai.decision import build_decision_payload
from sentinelxai.explainability import humanize_contributions

from .service import get_service

st.set_page_config(
    page_title="SentinelXAI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- theme -----------------------------------------------------------------
PRIMARY = "#2ECC71"
DANGER = "#E74C3C"
WARN = "#F1C40F"


def _banner(svc) -> bool:
    if not svc["available"]:
        st.warning(
            "⚠️ Trained model not found. Run `python scripts/train_final_lightgbm.py` "
            "(after `make data`) to enable predictions. Pages show static info only."
        )
        return False
    return True


def _predict(svc, record: dict):
    """Single-record prediction + explanation + decision, in-process."""
    result = svc["inference"].predict_single(record)
    pc = result.predicted_classes[0]
    conf = result.confidences[0]
    proba = result.probabilities[0]
    explanation = None
    human = []
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
    return pc, conf, proba, explanation, decision


# --- pages -----------------------------------------------------------------


def page_dashboard(svc):
    st.title("🛡 SentinelXAI — SOC Dashboard")
    hist = svc["store"].get_history(limit=500)
    if hist:
        df = pd.DataFrame([h.to_dict() for h in hist])
        total = len(df)
        critical = int((df["risk"] == "CRITICAL").sum())
        high = int((df["risk"] == "HIGH").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts", total)
        c2.metric("Critical", critical, delta_color="inverse")
        c3.metric("High", high)
        c4.metric("Model", "ONLINE" if svc["available"] else "OFFLINE")
        st.subheader("Attack Distribution")
        dist = df["predicted_class"].value_counts()
        st.bar_chart(dist)
        st.subheader("Recent Alerts")
        st.dataframe(df[["timestamp", "predicted_class", "confidence", "risk"]], use_container_width=True)
    else:
        st.info("No predictions logged yet. Use Live Prediction to generate alerts.")


def page_predict(svc):
    st.title("🔍 Live Prediction")
    if not _banner(svc):
        return
    mode = st.radio("Input mode", ["JSON record", "CSV upload"], horizontal=True)
    if mode == "JSON record":
        sample = '{\n  "Flow Duration": 0.0,\n  "SYN Flag Count": 1.0\n}'
        text = st.text_area("Engineered feature record (JSON)", value=sample, height=200)
        if st.button("Predict"):
            try:
                import json

                record = json.loads(text)
                pc, conf, proba, expl, dec = _predict(svc, record)
                _render_prediction(pc, conf, proba, dec, expl)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid input: {exc}")
    else:
        file = st.file_uploader("Upload CSV of engineered features", type="csv")
        if file and st.button("Predict batch"):
            df = pd.read_csv(file)
            records = df.to_dict(orient="records")
            out = []
            for rec in records:
                pc, conf, proba, expl, dec = _predict(svc, rec)
                out.append({"predicted_class": pc, "confidence": round(conf, 4), "risk": dec["risk"]})
            st.dataframe(pd.DataFrame(out), use_container_width=True)


def _render_prediction(pc, conf, proba, dec, expl):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Prediction: {pc}")
        st.metric("Confidence", f"{conf:.1%}")
        st.metric("Risk", dec["risk"])
        st.caption(f"Confidence band: {dec['confidence_band']}")
        if dec["is_failure_candidate"]:
            st.error("⚠ Low-confidence prediction — treat with caution.")
        st.subheader("Recommended Investigation")
        for r in dec["recommendations"]:
            st.markdown(f"- {r}")
    with col2:
        st.subheader("Why? (SHAP)")
        if expl is not None:
            for sent in expl.human_explanation or humanize_contributions(expl):
                st.markdown(f"- {sent}")
            chart = pd.DataFrame(
                [{"feature": c.name, "SHAP": c.value} for c in expl.top_contributions]
            )
            st.bar_chart(chart.set_index("feature")["SHAP"])
        else:
            st.info("SHAP explanations require the trained model.")


def page_explain(svc):
    st.title("🧠 Explainable AI")
    if not _banner(svc):
        return
    st.info("Paste an engineered-feature record to see local SHAP explanations and global importance.")
    import json

    text = st.text_area("Engineered feature record (JSON)", height=150)
    if st.button("Explain"):
        try:
            record = json.loads(text)
            pc, conf, proba, expl, dec = _predict(svc, record)
            _render_prediction(pc, conf, proba, dec, expl)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid input: {exc}")


def page_studio(svc):
    st.title("🎛 Decision Intelligence Studio")
    if not _banner(svc):
        return
    feats = svc["inference"].feature_names
    editable = [f for f in ["SYN Flag Count", "Flow Duration", "Destination Port", "Packet Length Mean", "Bwd Packet Length Mean"] if f in feats]
    st.caption("Modify features and watch the prediction, confidence, and risk update live.")
    record = {}
    cols = st.columns(len(editable))
    for i, f in enumerate(editable):
        record[f] = cols[i].slider(f, 0.0, 10000.0, 0.0, key=f)
    # Fill remaining features with zeros so predict gets a complete record.
    full = {f: record.get(f, 0.0) for f in feats}
    pc, conf, proba, expl, dec = _predict(svc, full)
    st.subheader(f"Live Prediction: {pc}")
    st.metric("Confidence", f"{conf:.1%}")
    st.metric("Risk", dec["risk"])
    if expl is not None:
        chart = pd.DataFrame([{"feature": c.name, "SHAP": c.value} for c in expl.top_contributions])
        st.bar_chart(chart.set_index("feature")["SHAP"])


def page_failure(svc):
    st.title("🩺 Failure Explorer")
    st.caption("Surfaces low-confidence predictions (failure candidates) and their explanations.")
    hist = svc["store"].get_history(limit=500)
    failures = [h for h in hist if (h.confidence or 1.0) < svc["decision_cfg"].modulation.failure_confidence_threshold]
    if not failures:
        st.info("No low-confidence predictions logged yet.")
        return
    sel = st.selectbox(
        "Select a failure candidate",
        [f"#{h.id} {h.predicted_class} ({h.confidence:.2f})" for h in failures],
    )
    selected_id = int(sel.split(" ")[0][1:])
    idx = next(h for h in failures if h.id == selected_id)
    st.write(idx.to_dict())


def page_analytics(svc):
    st.title("📊 Model Analytics")
    metrics_path = svc["cfg"].paths.reports_dir / "lightgbm" / "lightgbm_metrics.json"
    if metrics_path.exists():
        import json

        m = json.loads(metrics_path.read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
        c2.metric("Macro F1", f"{m.get('f1_macro', 0):.4f}")
        c3.metric("Weighted F1", f"{m.get('f1_weighted', 0):.4f}")
    else:
        st.info("Train the final model to populate metrics.")
    cm = svc["cfg"].paths.reports_dir / "lightgbm" / "figures" / "lightgbm_confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption="Confusion Matrix (validation)")


def page_about(svc):
    st.title("ℹ️ About / Architecture")
    st.markdown(
        """
        **SentinelXAI** — Explainable AI Decision Intelligence Platform for Network
        Intrusion Detection.

        Pipeline: `Network Flow → Validation → LightGBM → TreeSHAP → Decision
        Intelligence → Analyst Dashboard`.

        - Model: LightGBM (Macro F1 0.9201 on validation)
        - Explainability: TreeSHAP
        - Decision layer: confidence, risk policy, recommendations
        - Backend: FastAPI · Frontend: Streamlit · DB: SQLite · Deploy: Docker
        """
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Live Prediction": page_predict,
    "Explainable AI": page_explain,
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
