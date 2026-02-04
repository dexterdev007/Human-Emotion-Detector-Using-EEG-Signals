#!/usr/bin/env python3
"""
Generate web/model.js with trained regression coefficients.
No backend required for inference; the browser will run predictions.
This script is intentionally simple and self-contained.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


CHANNEL_LABELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

EMOTION_LABELS = ["Calm", "Sad", "Fearful", "Angry", "Surprised", "Happy"]


def load_eeg_data(dataset_path: Path, max_subjects: int = 5) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    files = sorted([f for f in dataset_path.iterdir() if f.name.startswith("s") and f.suffix == ".csv"])
    if not files:
        raise FileNotFoundError("No EEG CSV files found in dataset folder.")

    files = files[:max_subjects]
    all_data = []

    print(f"Found {len(files)} subject files:")
    for file in files[:10]:
        print(f"  - {file.name}")

    for file in files:
        data = pd.read_csv(file, header=None)
        data.columns = [f"EEG_Ch_{i + 1}" for i in range(19)]
        all_data.append(data)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined


def create_synthetic_labels(X: np.ndarray) -> np.ndarray:
    # Create deterministic synthetic labels based on weighted EEG channels
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, X.shape[1])
    weights[[0, 4, 9, 14]] *= 2.0
    scores = X @ weights
    scores = (scores - scores.min()) / (scores.max() - scores.min()) * 10
    return scores


def create_multiclass_labels(y_cont: np.ndarray, n_classes: int) -> np.ndarray:
    quantiles = np.quantile(y_cont, np.linspace(0, 1, n_classes + 1))
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-6
    return np.digitize(y_cont, quantiles[1:-1], right=True)


def main() -> None:
    np.random.seed(42)

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "dataset"

    data = load_eeg_data(dataset_path, max_subjects=5)
    eeg_columns = [f"EEG_Ch_{i + 1}" for i in range(19)]

    # Fill missing values with mean
    data[eeg_columns] = data[eeg_columns].fillna(data[eeg_columns].mean())

    example_values = data[eeg_columns].iloc[0].round(4).tolist()
    sample_pool = (
        data[eeg_columns]
        .sample(n=200, random_state=42)
        .round(4)
        .values.tolist()
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[eeg_columns].values)

    # Synthetic labels
    y_cont = create_synthetic_labels(X_scaled)

    # Train models
    linear_model = LinearRegression()
    linear_model.fit(X_scaled, y_cont)

    y_multi = create_multiclass_labels(y_cont, len(EMOTION_LABELS))
    logistic_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    logistic_model.fit(X_scaled, y_multi)

    model_payload = {
        "channelLabels": CHANNEL_LABELS,
        "exampleValues": example_values,
        "sampleValues": sample_pool,
        "emotionLabels": EMOTION_LABELS,
        "model": {
            "scalerMean": scaler.mean_.tolist(),
            "scalerScale": scaler.scale_.tolist(),
            "linearCoef": linear_model.coef_.tolist(),
            "linearIntercept": float(linear_model.intercept_),
            "logisticCoef": logistic_model.coef_.tolist(),
            "logisticIntercept": logistic_model.intercept_.tolist(),
        },
    }

    output_path = project_root / "web" / "model.js"
    output_path.write_text(
        "window.APP_CONFIG = "
        + json.dumps(model_payload, indent=2)
        + ";\n",
        encoding="utf-8",
    )

    print(f"Model file written to {output_path}")


if __name__ == "__main__":
    main()
