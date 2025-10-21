# 🛍 Smart Pricing: A Multimodal Price Prediction Engine

This project implements a sophisticated machine learning pipeline to predict product prices using multimodal catalog data, including textual descriptions, numerical features, and image embeddings. The methodology is structured as a three-stage stacked ensemble to maximize predictive accuracy.

---

## 📜 Project Overview

The primary goal of this project is to accurately predict product prices by leveraging a rich dataset of catalog information.

While traditional models rely on a single data type, this approach combines **text**, **image**, and **structured numerical data** to create a more robust and accurate price predictor.

This repository contains the complete pipeline, from a simple text-based baseline to a final, high-performance stacked ensemble model.

---

## ⚙️ Methodology & Pipeline Architecture

The solution is built using a **three-stage pipeline**, where each stage builds upon the last to improve performance.

### Stage 1: Text-Only Baseline Model

The first stage establishes a performance baseline using only the textual content of the product catalog.

- **Data Source**: Uses only the `catalog_content` textual field.
- **Feature Extraction**: Text is vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency).
  - Top 10,000 features (unigrams + bigrams).
- **Modeling**: A **Ridge Regression** model (α = 1.0) is trained on TF-IDF features.
- **Validation**: Evaluated using 5-fold cross-validation.
- **Result**: Baseline SMAPE = **51.3%**

---

### Stage 2: Multimodal Enhancement

This stage enhances the model by integrating text, image, and numerical features.

#### 🔹 Feature Engineering

**Text Features:**
- TF-IDF vectors from Stage 1 reduced to **128 dimensions** using Truncated SVD.

**Image Features:**
- Precomputed CNN embeddings are used for image representation.
- Reduced to **64 principal components** using PCA.

**Numeric Features:**
- Includes pack quantity (`ipq`), text length, word counts, and binary flags (e.g., currency presence, image availability).
- All standardized using `StandardScaler`.

#### 🔹 Modeling & Validation

- **Model**: LightGBM trained on the combined multimodal feature set.
- **Validation**: 5-fold Stratified K-Fold for robust evaluation.
- **Result**: Multimodal SMAPE = **49.6%**

---

### Stage 3: Stacked Ensemble for Final Inference

The final stage combines predictions from the baseline and multimodal models to produce the final price prediction.

#### Approach

A meta-model (**Ridge Regression**) learns how to optimally blend the Stage 1 and Stage 2 predictions.

#### Process

1. Out-of-fold predictions from baseline (`test_baseline_log.npy`) and multimodal (`test_multimodal_log.npy`) models are loaded.
2. These two sets of predictions form a new meta-dataset (2 features).
3. A final Ridge Regression meta-model is trained to combine them.

#### Result

- **Final SMAPE: 48.2%**

This stacked ensemble yields the best overall performance.

---

## 📊 Performance Evaluation

All models were evaluated using the **Symmetric Mean Absolute Percentage Error (SMAPE)** metric.

| Model | Features Used | Algorithm | Final SMAPE (%) | Improvement |
|-------|---------------|-----------|-----------------|-------------|
| **Baseline** | TF-IDF (Text Only) | Ridge Regression | 51.3 | — |
| **Multimodal** | TF-IDF + SVD + Image + Numeric | LightGBM | 49.6 | −1.7 % |
| **Final Stack** | Baseline + Multimodal Predictions | Ridge (Meta-model) | 48.2 | −1.4 % |
| **Total Improvement** | | | | **−3.1 %** |

This stepwise architecture demonstrates a **total improvement of 3.1%** over the baseline, validating the power of multimodal learning and ensemble modeling.

---

## 💡 Key Findings

✅ **Multimodal Feature Integration:**  
Combining text, image, and numerical data provided a richer predictive signal than text alone.

✅ **Ensemble Learning:**  
Stacking the Ridge and LightGBM models effectively reduced overall error.

✅ **Dimensionality Reduction:**  
Using Truncated SVD and PCA managed high-dimensional data efficiently.

✅ **Model Selection:**  
Simple yet efficient models — Ridge Regression for lightweight stages, LightGBM for complex multimodal learning — gave a strong balance between speed and accuracy.

---

## 🚀 How to Run

### Prerequisites

- Python ≥ 3.8
- pip

### Installation
```bash
git clone https://github.com/your-username/smart-pricing.git
cd smart-pricing
pip install -r requirements.txt
```

`requirements.txt` should include:
```
numpy
pandas
scikit-learn
lightgbm
```

### Directory Structure
```
.
├── data/
│   ├── catalog_data.csv        # Main data file
│   └── image_embeddings.npy    # Precomputed CNN embeddings
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/
│   ├── 1_baseline_model.py
│   ├── 2_multimodal_model.py
│   └── 3_stacking_model.py
├── requirements.txt
└── README.md
```

### Running the Pipeline

**1️⃣ Run the baseline model:**
```bash
python scripts/1_baseline_model.py
```

Generates `test_baseline_log.npy`

**2️⃣ Run the multimodal model:**
```bash
python scripts/2_multimodal_model.py
```

Generates `test_multimodal_log.npy`

**3️⃣ Run the final stacking ensemble:**
```bash
python scripts/3_stacking_model.py
```

Produces final output: `test_out_final.csv`

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
