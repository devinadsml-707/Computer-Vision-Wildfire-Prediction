# CNN Wildfire Risk Predictor

A deep learning project that classifies satellite images to identify areas at risk of wildfire using Convolutional Neural Networks (CNN), including transfer learning with ResNet50.

🚀 [Live Demo on Hugging Face](https://huggingface.co/spaces/Devina707/CNN-Wildfire-Predictor#wildfire-predictor) · 📓 [View Analysis Notebook](notebooks/01_wildfire_cnn_analysis.ipynb) · 📥 [Download Model & Inference Data](https://drive.google.com/drive/folders/1K9wB9y_IPh616RsMxPlknGJ1lFtFh99x?usp=sharing)

---

## Problem Background

Wildfires have become one of the most destructive environmental disasters, escalating in recent years due to climate change. Canada alone — holding 9% of the world's forests — experienced its worst wildfire season in 2023, with over 18 million hectares burned. Traditional detection methods relying on ground-based observations are limited and do not provide real-time updates.

Satellite remote sensing offers continuous monitoring across all areas, including remote regions. By training CNN models on satellite imagery, it becomes possible to develop applications that continuously predict wildfire risk — enabling faster preventive action and early warning systems.

---

## Project Output

- **CNN Model**: A binary image classifier distinguishing wildfire-risk areas from safe areas, achieving **92.59% test accuracy**
- **Deployed App**: Interactive wildfire risk prediction app hosted on Hugging Face Spaces

**Model Comparison:**

| Model | Architecture | Accuracy | Val Loss | AUC | Recall | FN |
|-------|-------------|----------|----------|-----|--------|----|
| ANN1 | Basic CNN (3 conv blocks) | 53.25% | 0.6962 | 0.9913 | 0.98 | 61 |
| ANN2 | Improved CNN (4 conv blocks + BatchNorm + Dropout + augmentation) | 47.43% | 0.6925 | 0.9965 | 0.99 | 47 |
| **ANN3** | **ResNet50 Transfer Learning ✅** | **92.59%** | **0.1858** | **0.9793** | **0.92** | **258** |

> **Best Model: ANN3 (ResNet50)** — selected for its significantly lower val loss and best fit (smallest gap between train and validation loss), indicating strong generalization with no overfitting.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_wildfire_cnn_analysis.ipynb    # Full analysis: EDA, modeling, evaluation, conclusions
│   └── 02_inference.ipynb                # Model inference and prediction testing
├── requirements.txt
└── README.md
```

> 📥 Model artifacts (`.keras`, `.pkl`) and inference data are hosted on [Google Drive](https://drive.google.com/drive/folders/1K9wB9y_IPh616RsMxPlknGJ1lFtFh99x?usp=sharing) — download and place in a `models/` folder before running `02_inference.ipynb`.

---

## Data

| Detail | Info |
|--------|------|
| **Source** | [Wildfire Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) |
| **Location** | Canada (holds 9% of the world's forests) |
| **Total Images** | 42,850 satellite images |
| **Classes** | `wildfire` (22,710 images, 53%) · `nowildfire` (20,140 images, 47%) |
| **Image Size** | 350×350px (resized to 128×128 for ANN1/ANN2, 224×224 for ANN3) |
| **Split** | 70% train / 15% validation / 15% test (stratified) |
| **Class Balance** | Balanced — no resampling required |

---

## Method

### Models Trained

**ANN1 — Basic CNN**
- 3 convolutional blocks, no regularization, no augmentation
- Used as baseline

**ANN2 — Improved CNN**
- 4 convolutional blocks + BatchNorm + Dropout
- Data augmentation on training set (flip, rotation, zoom, shift, shear)

**ANN3 — ResNet50 Transfer Learning (Best Model)**
- Two-phase training: Phase 1 freezes ResNet50 base to train classification head; Phase 2 fine-tunes with low learning rate (1e-5) to prevent catastrophic forgetting of ImageNet knowledge
- Augmentation applied on training set (same as ANN2, excluding brightness augmentation to preserve ImageNet feature distribution)

### Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix (TP, TN, FP, FN)

> **Note on FN vs FP**: False Negatives (predicting no wildfire when there is one) are prioritized as the more critical error — missing a wildfire risk poses far greater environmental and financial risk than a false alarm.

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/cnn-wildfire-predictor.git
cd cnn-wildfire-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model artifacts**

Download from [Google Drive](https://drive.google.com/drive/folders/1K9wB9y_IPh616RsMxPlknGJ1lFtFh99x?usp=sharing) and place in `models/` directory.

**4. Run inference notebook**
```
notebooks/02_inference.ipynb
```

> ⚠️ This project was trained on Google Colab due to GPU requirements. Local training is possible but may be slow without a GPU.

---

## Stacks

| Category | Libraries |
|----------|-----------|
| **Data & Analysis** | pandas, numpy |
| **Image Processing** | OpenCV (`cv2`), Pillow (`PIL`), glob |
| **Visualization** | matplotlib, seaborn |
| **Modeling** | TensorFlow, Keras (Sequential API, ResNet50) |
| **Evaluation** | scikit-learn (classification report, confusion matrix, ROC-AUC) |
| **Utilities** | pickle, json, shutil, os, pathlib |

---

## References

- **Dataset**: [Wildfire Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
- **Deployed App**: [Hugging Face Spaces](https://huggingface.co/spaces/Devina707/CNN-Wildfire-Predictor#wildfire-predictor)
- **Model & Inference Files**: [Google Drive](https://drive.google.com/drive/folders/1K9wB9y_IPh616RsMxPlknGJ1lFtFh99x?usp=sharing)

---

*Built by Devina Agustina*
