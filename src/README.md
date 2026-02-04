# Human Emotion Detector Using EEG Signals (Simple ML)

This folder contains a **single, easy-to-understand** Python script that
implements the full EEG pipeline with only **linear regression** and
**logistic regression**.

## File Structure (src/)
```
src/
├── main.py          # Simple end-to-end ML pipeline
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## How to Run
```bash
python3 src/main.py
```

## What main.py Does
### 1. Dataset Loading and Exploration
- Loads EEG CSV files from `dataset/`
- Prints dataset stats and sample rows

### 2. Data Preprocessing
- Fills missing values with channel means
- Normalizes features with `StandardScaler`
- Creates synthetic emotion labels (continuous + binary)
- Splits into train/test sets

### 3. Linear Regression Analysis
- Predicts emotion intensity (0–10)
- Reports MSE and R²
- Shows top feature coefficients

### 4. Logistic Regression Analysis
- Classifies emotion as High vs Low
- Reports accuracy, confusion matrix, and classification report

### 5. Model Comparison and Discussion
- Compares both models and prints simple limitations
- No accounting for individual subject variability

## Future Improvements
- Implement actual multi-reservoir ESN models
- Add frequency domain features (FFT, wavelets)
- Include subject-specific model training
- Real emotion labels from experimental data

## Output Example
```
🧠 EEG-BASED EMOTION ANALYSIS USING CLASSICAL MACHINE LEARNING
🎓 Academic Assignment - Inspired by Neurocomputing 2025 Research
================================================================================

STEP 1: DATASET LOADING AND EXPLORATION
============================================================
Found 36 subject files:
   1. s00.csv
   2. s01.csv
   ...
   5. s04.csv

📊 Combined Dataset:
   Shape: (155005, 20)
   Subjects: 5
   EEG Channels: 19
   Total Samples: 155005

🎯 Linear Regression Results:
   Training MSE: 2.1234
   Testing MSE:  2.1456
   Training R²:  0.7890
   Testing R²:   0.7856

🎯 Logistic Regression Results:
   Training Accuracy: 0.8234
   Testing Accuracy:  0.8198
```

## Author
EEG Emotion Analysis Project
Academic Assignment - Machine Learning for Neuroscience

## License
This project is for educational purposes only.
