"""Deterministic human-readable explanation text (Phase 4).

Turns SHAP contributions into analyst-facing sentences. The mapping is fully
deterministic and template-based — it never invents cybersecurity advice, only
translates signed SHAP values into plain language about the *predicted* class
(per CLAUDE.md "do not generate random explanations").
"""

from __future__ import annotations

from .shap_engine import LocalExplanation

# Friendly display names for the most common CICIDS2017 flow features. Any
# feature not listed falls back to its raw (title-cased) name, so this map is
# an enhancement, never a required lookup.
FEATURE_LABELS: dict[str, str] = {
    "Flow Duration": "flow duration",
    "Total Fwd Packets": "total forward packets",
    "Total Backward Packets": "total backward packets",
    "Total Length of Fwd Packets": "total forward bytes",
    "Total Length of Bwd Packets": "total backward bytes",
    "Fwd Packet Length Max": "max forward packet length",
    "Fwd Packet Length Mean": "mean forward packet length",
    "Bwd Packet Length Max": "max backward packet length",
    "Bwd Packet Length Mean": "mean backward packet length",
    "Flow Bytes/s": "flow bytes per second",
    "Flow Packets/s": "flow packets per second",
    "Flow IAT Mean": "mean inter-arrival time",
    "Flow IAT Std": "std of inter-arrival time",
    "Fwd IAT Total": "total forward inter-arrival time",
    "Bwd IAT Total": "total backward inter-arrival time",
    "SYN Flag Count": "SYN flag count",
    "ACK Flag Count": "ACK flag count",
    "PSH Flag Count": "PSH flag count",
    "RST Flag Count": "RST flag count",
    "FIN Flag Count": "FIN flag count",
    "URG Flag Count": "URG flag count",
    "Init_Win_bytes_forward": "initial forward TCP window",
    "Init_Win_bytes_backward": "initial backward TCP window",
    "act_data_pkt_fwd": "active data packets forward",
    "Destination Port": "destination port",
    "Protocol": "protocol",
    "Subflow Fwd Packets": "forward subflow packets",
    "Subflow Bwd Packets": "backward subflow packets",
    "Idle Mean": "mean idle time",
    "Idle Std": "std of idle time",
}


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.strip())


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def humanize_contributions(
    explanation: LocalExplanation, *, max_sentences: int | None = None
) -> list[str]:
    """Return plain-language sentences describing the top SHAP contributors.

    Sentences are ordered by contribution rank. A negative SHAP value means the
    feature pushed the probability of the predicted class *down*; the wording
    reflects that direction literally rather than sign-flipping the class.
    """
    limit = max_sentences if max_sentences else len(explanation.top_contributions)
    sentences: list[str] = []
    for c in explanation.top_contributions[:limit]:
        label = _label(c.name)
        if c.direction == "increase":
            sentence = (
                f"Elevated {label} (SHAP {_fmt(c.value)}) increased the model's "
                f"estimated probability of {explanation.predicted_class}."
            )
        else:
            sentence = (
                f"Reduced {label} (SHAP {_fmt(c.value)}) lowered the estimated "
                f"probability of {explanation.predicted_class}."
            )
        sentences.append(sentence)
    return sentences


def summarize(explanation: LocalExplanation) -> str:
    """One-line summary naming the single most influential feature."""
    if not explanation.top_contributions:
        return f"Prediction: {explanation.predicted_class} (no strong feature signal)."
    top = explanation.top_contributions[0]
    return (
        f"Prediction of {explanation.predicted_class} is most strongly driven by "
        f"{_label(top.name)} (SHAP {_fmt(top.value)})."
    )
