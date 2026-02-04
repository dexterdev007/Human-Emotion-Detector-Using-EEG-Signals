"""
Human Emotion Detector Using EEG Signals
Simple, easy-to-understand ML pipeline (Linear + Logistic Regression)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
MAX_SUBJECTS = 5
N_CHANNELS = 19


def print_section(title: str, bullets: list) -> None:
    print(f"### {title}")
    for item in bullets:
        print(f"- {item}")


def load_eeg_data(dataset_dir: Path, max_subjects: int = 5) -> pd.DataFrame:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    files = sorted([f for f in dataset_dir.iterdir() if f.name.startswith("s") and f.suffix == ".csv"])
    if not files:
        raise FileNotFoundError("No EEG CSV files found in dataset folder.")

    files = files[:max_subjects]
    all_data = []

    print(f"Found {len(files)} subject files:")
    for file in files:
        print(f"  - {file.name}")

    for file in files:
        df = pd.read_csv(file, header=None)
        df.columns = [f"EEG_Ch_{i + 1}" for i in range(N_CHANNELS)]
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def main():
    # ==============================
    # 1. Dataset Loading and Exploration
    # ==============================
    print_section(
        "1. Dataset Loading and Exploration",
        [
            "Loads EEG data from multiple subject files",
            "Displays dataset statistics and sample data",
            "Provides comprehensive data overview",
        ],
    )
    data = load_eeg_data(DATASET_DIR, max_subjects=MAX_SUBJECTS)
    print(f"Shape: {data.shape}")
    print("Sample rows:")
    print(data.head(3))
    print("\nBasic statistics:")
    print(data.describe().loc[["mean", "std", "min", "max"]].round(3))

    # ==============================
    # 2. Data Preprocessing
    # ==============================
    print_section(
        "\n2. Data Preprocessing",
        [
            "Handles missing values using channel means",
            "Normalizes features using StandardScaler",
            "Creates synthetic emotion labels (continuous and binary)",
            "Splits data into training and testing sets",
        ],
    )
    eeg_cols = [f"EEG_Ch_{i + 1}" for i in range(N_CHANNELS)]

    # Fill missing values with column means
    data[eeg_cols] = data[eeg_cols].fillna(data[eeg_cols].mean())
    print("- Missing values handled with channel means")

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(data[eeg_cols].values)
    print("- Features normalized using StandardScaler")

    # Create synthetic labels (continuous + binary)
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, N_CHANNELS)
    weights[[0, 4, 9, 14]] *= 2.0
    scores = X @ weights
    y_cont = (scores - scores.min()) / (scores.max() - scores.min()) * 10
    median_val = np.median(y_cont)
    y_bin = (y_cont > median_val).astype(int)
    print("- Synthetic emotion labels created (continuous and binary)")

    # Split data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cont, test_size=0.2, random_state=42
    )
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    print("- Data split into train and test sets")

    # ==============================
    # 3. Linear Regression Analysis
    # ==============================
    print_section(
        "\n3. Linear Regression Analysis",
        [
            "Predicts continuous emotion intensity scores (0-10 scale)",
            "Evaluates using Mean Squared Error (MSE) and R\u00b2",
            "Displays feature importance based on coefficients",
        ],
    )
    lin_model = LinearRegression()
    lin_model.fit(X_train_c, y_train_c)

    y_pred = lin_model.predict(X_test_c)
    mse = mean_squared_error(y_test_c, y_pred)
    r2 = r2_score(y_test_c, y_pred)
    print(f"- MSE: {mse:.4f}")
    print(f"- R²: {r2:.4f}")

    # Feature importance (absolute coefficients)
    coef_importance = np.abs(lin_model.coef_)
    top_idx = np.argsort(coef_importance)[-5:][::-1]
    print("- Top 5 Features (by coefficient magnitude):")
    for idx in top_idx:
        print(f"  EEG_Ch_{idx + 1}: {coef_importance[idx]:.4f}")

    # ==============================
    # 4. Logistic Regression Analysis
    # ==============================
    print_section(
        "\n4. Logistic Regression Analysis",
        [
            "Classifies emotions as High vs Low",
            "Evaluates using accuracy and confusion matrix",
            "Provides classification report with precision, recall, F1-score",
        ],
    )
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_b, y_train_b)

    y_pred_b = log_model.predict(X_test_b)
    acc = accuracy_score(y_test_b, y_pred_b)
    cm = confusion_matrix(y_test_b, y_pred_b)
    print(f"- Accuracy: {acc:.4f}")
    print("- Confusion Matrix:")
    print(cm)
    print("- Classification Report:")
    print(classification_report(y_test_b, y_pred_b, target_names=["Low", "High"]))

    # ==============================
    # 5. Model Comparison and Discussion
    # ==============================
    print_section(
        "\n5. Model Comparison and Discussion",
        [
            "Compares performance of both models",
            "Discusses limitations of classical regression approaches",
            "References advanced multi-reservoir ESN methods",
        ],
    )
    print(f"- Linear Regression (MSE): {mse:.4f}")
    print(f"- Logistic Regression (Accuracy): {acc:.4f}")
    print("- Limitations: linear models may miss non-linear EEG patterns.")
    print("- Advanced methods (e.g., reservoir computing) could improve results.")


if __name__ == "__main__":
    main()
