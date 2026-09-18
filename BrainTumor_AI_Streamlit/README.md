# BrainTumor AI — Streamlit App

A polished Streamlit interface for the Brain MRI binary-classification project.

## Features

- Dark navy + blue professional medical-AI UI
- MRI image upload and preview
- VGG16 inference
- Tumor / No Tumor probabilities
- Model information and comparison charts
- CNN confusion matrix from the original notebook
- How It Works page
- Limitations & medical disclaimer
- GitHub and LinkedIn contact links
- Downloadable prediction report
- Responsive Streamlit layout

## Model

The current app is configured for the recorded VGG16 transfer-learning model.

Recorded test accuracy from the original project:
- Custom CNN: 87.56%
- VGG16: 91.56%

These are single-run test-split results from the supplied notebook and are not clinical validation.

## Important: model artifact

The GitHub repository currently contains the notebook, README, dataset folder and presentation, but does not contain the trained `.keras` model files.

Place:

`models/vgg16_model_final.keras`

in this folder before running the app.

See `models/MODEL_SETUP.txt`.

## Run locally

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Notes

The original training workflow used:
- 128 × 128 RGB
- pixel scaling by 255
- VGG16 ImageNet backbone
- frozen VGG16 feature extractor
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dropout(0.3)
- 2-class softmax

The app intentionally keeps the inference preprocessing consistent with that experiment.

## Medical disclaimer

This is an educational/research project. It is not a medical device and does not provide a medical diagnosis.
Clinical interpretation must be performed by a qualified healthcare professional.
