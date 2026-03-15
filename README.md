# Neuro AI Suite (EEG + MRI)

Repository is organized into 3 folders:

1. `human emotion recognition/`
2. `brain tumor classification/`
3. `web dev/`

## 1) Human Emotion Recognition

- Notebook: `human emotion recognition/Human_Emotion_Recognition.ipynb`
- Dataset: `human emotion recognition/dataset/emotions.csv`

## 2) Brain Tumor Classification

- Notebook: `brain tumor classification/Brain_Tumor_Classification.ipynb`
- Dataset folder: `brain tumor classification/dataset/`

Expected Kaggle dataset path after download/unzip:

`brain tumor classification/dataset/brain-tumor-mri-dataset/Training`

## 3) Web Dev

- Frontend: `web dev/frontend/`
- Backend: `web dev/backend/run_server.py`

Run web app:

```bash
python "web dev/backend/run_server.py"
```

Open:

- `http://127.0.0.1:5500/index.html`
- `http://127.0.0.1:5500/eeg.html`
- `http://127.0.0.1:5500/brain-tumor.html`
