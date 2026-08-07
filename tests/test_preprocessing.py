from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelxai.config import LinearModelPreprocessingConfig, PortBucketRule
from sentinelxai.models.preprocessing import (
    apply_scaler,
    bucket_destination_port,
    build_linear_model_features,
    compute_sample_weights,
    downcast_numeric_columns,
    encode_labels,
    fit_label_encoder,
    fit_scaler,
    log1p_transform,
)


@pytest.fixture
def lmp_config() -> LinearModelPreprocessingConfig:
    return LinearModelPreprocessingConfig(
        destination_port_buckets=(
            PortBucketRule(name="port_well_known", min_port=None, max_port=1023),
            PortBucketRule(name="port_registered", min_port=1024, max_port=49151),
            PortBucketRule(name="port_dynamic", min_port=49152, max_port=None),
        ),
        destination_port_indicator_ports=(22, 80),
        log1p_all_features=True,
    )


# --- memory management ---


def test_downcast_numeric_columns_converts_to_float32():
    df = pd.DataFrame({"A": np.array([1.0, 2.0], dtype=np.float64), "B": [1, 2]})
    result = downcast_numeric_columns(df, ["A", "B"])
    assert result["A"].dtype == np.float32
    assert result["B"].dtype == np.float32


def test_downcast_numeric_columns_preserves_values_within_float32_precision():
    df = pd.DataFrame({"A": [1.5, 2.25, 100000.0]})
    result = downcast_numeric_columns(df, ["A"])
    np.testing.assert_allclose(result["A"], df["A"], rtol=1e-6)


def test_downcast_numeric_columns_skips_non_numeric_columns():
    df = pd.DataFrame({"A": [1.0, 2.0], "Label": ["BENIGN", "DDoS"]})
    result = downcast_numeric_columns(df, ["A", "Label"])
    assert result["Label"].dtype == object  # untouched


def test_downcast_numeric_columns_does_not_mutate_input():
    df = pd.DataFrame({"A": np.array([1.0, 2.0], dtype=np.float64)})
    original_dtype = df["A"].dtype
    downcast_numeric_columns(df, ["A"])
    assert df["A"].dtype == original_dtype


# --- label encoding ---


def test_fit_and_encode_labels_round_trip():
    y_train = pd.Series(["BENIGN", "DDoS", "BENIGN", "Heartbleed"])
    encoder = fit_label_encoder(y_train)
    encoded = encode_labels(encoder, y_train)
    assert set(encoded) == {0, 1, 2}
    assert len(encoder.classes_) == 3


def test_encode_labels_raises_on_unseen_label():
    encoder = fit_label_encoder(pd.Series(["BENIGN", "DDoS"]))
    with pytest.raises(ValueError, match="not present in the encoder"):
        encode_labels(encoder, pd.Series(["BENIGN", "Heartbleed"]))


def test_encode_labels_val_uses_train_fitted_encoder():
    """The exact leakage-safety contract: encoder fit on train, applied to val."""
    y_train = pd.Series(["BENIGN", "DDoS", "Heartbleed"])
    y_val = pd.Series(["DDoS", "BENIGN"])  # subset of train's classes
    encoder = fit_label_encoder(y_train)
    encoded_val = encode_labels(encoder, y_val)
    assert list(encoder.inverse_transform(encoded_val)) == ["DDoS", "BENIGN"]


# --- destination port bucketing ---


def test_bucket_destination_port_assigns_correct_bucket(lmp_config):
    df = pd.DataFrame({"Destination Port": [80, 22, 8080, 60000], "Other": [1, 2, 3, 4]})
    result = bucket_destination_port(df, "Destination Port", lmp_config)
    assert list(result["port_well_known"]) == [1, 1, 0, 0]
    assert list(result["port_registered"]) == [0, 0, 1, 0]
    assert list(result["port_dynamic"]) == [0, 0, 0, 1]


def test_bucket_destination_port_indicator_dummies(lmp_config):
    df = pd.DataFrame({"Destination Port": [80, 22, 443], "Other": [1, 2, 3]})
    result = bucket_destination_port(df, "Destination Port", lmp_config)
    assert list(result["port_is_22"]) == [0, 1, 0]
    assert list(result["port_is_80"]) == [1, 0, 0]


def test_bucket_destination_port_drops_raw_column(lmp_config):
    df = pd.DataFrame({"Destination Port": [80], "Other": [1]})
    result = bucket_destination_port(df, "Destination Port", lmp_config)
    assert "Destination Port" not in result.columns
    assert "Other" in result.columns


def test_bucket_destination_port_boundaries_are_inclusive(lmp_config):
    df = pd.DataFrame({"Destination Port": [1023, 1024, 49151, 49152], "Other": [0, 0, 0, 0]})
    result = bucket_destination_port(df, "Destination Port", lmp_config)
    assert list(result["port_well_known"]) == [1, 0, 0, 0]
    assert list(result["port_registered"]) == [0, 1, 1, 0]
    assert list(result["port_dynamic"]) == [0, 0, 0, 1]


# --- log1p ---


def test_log1p_transform_matches_numpy_directly():
    df = pd.DataFrame({"A": [0, 1, 10, 100]})
    result = log1p_transform(df, ["A"])
    np.testing.assert_allclose(result["A"], np.log1p([0, 1, 10, 100]))


def test_log1p_transform_raises_on_negative_values():
    df = pd.DataFrame({"A": [1, -5, 3]})
    with pytest.raises(ValueError, match="negative values"):
        log1p_transform(df, ["A"])


def test_log1p_transform_does_not_mutate_input():
    df = pd.DataFrame({"A": [0, 1, 10]})
    original = df.copy()
    log1p_transform(df, ["A"])
    pd.testing.assert_frame_equal(df, original)


# --- build_linear_model_features (integration of the two above) ---


def test_build_linear_model_features_combines_bucketing_and_log1p(lmp_config):
    df = pd.DataFrame({"Destination Port": [80], "Flow Duration": [99]})
    result = build_linear_model_features(df, "Destination Port", lmp_config)
    assert "Destination Port" not in result.columns
    assert "port_well_known" in result.columns
    assert np.isclose(result["Flow Duration"].iloc[0], np.log1p(99))


def test_build_linear_model_features_skips_log1p_on_sentinel_columns(lmp_config):
    """Regression test: Init_Win_bytes_forward legitimately holds -1 (a
    sentinel, see configs/features.yaml). A blind "log1p everything
    numeric" pass previously crashed on this with log1p(-1) == -inf.
    """
    df = pd.DataFrame(
        {"Destination Port": [80], "Flow Duration": [99], "Init_Win_bytes_forward": [-1]}
    )
    result = build_linear_model_features(
        df, "Destination Port", lmp_config, skip_log1p_columns=("Init_Win_bytes_forward",)
    )
    assert result["Init_Win_bytes_forward"].iloc[0] == -1  # untouched, not log1p'd
    assert np.isclose(result["Flow Duration"].iloc[0], np.log1p(99))  # other columns still transformed


# --- scaling ---


def test_fit_scaler_and_apply_scaler_train_has_zero_mean_unit_std():
    train_df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = fit_scaler(train_df, ["A"])
    scaled = apply_scaler(scaler, train_df, ["A"])
    assert np.isclose(scaled["A"].mean(), 0.0, atol=1e-10)
    assert np.isclose(scaled["A"].std(ddof=0), 1.0, atol=1e-10)


def test_apply_scaler_uses_train_fitted_parameters_on_val():
    """Leakage-safety contract: scaler fit on train stats, applied unchanged to val."""
    train_df = pd.DataFrame({"A": [10.0, 20.0, 30.0]})  # mean=20, std=~8.16
    val_df = pd.DataFrame({"A": [20.0]})  # exactly train's mean
    scaler = fit_scaler(train_df, ["A"])
    scaled_val = apply_scaler(scaler, val_df, ["A"])
    assert np.isclose(scaled_val["A"].iloc[0], 0.0, atol=1e-10)  # (20-20)/std == 0


# --- sample weights ---


def test_compute_sample_weights_upweights_rare_class():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # class 1 has 1/10 the frequency
    weights = compute_sample_weights(y)
    majority_weight = weights[y == 0][0]
    minority_weight = weights[y == 1][0]
    assert minority_weight > majority_weight
