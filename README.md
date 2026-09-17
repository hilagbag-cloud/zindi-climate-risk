# 🌍 Zindi Climate Risk and Health Prediction Challenge

Automated ML project setup designed for autonomous agent development with **Jules**.

---

## 🎯 Mission for Jules
1. Build and iterate on machine learning models to predict whether a recorded death is climate-sensitive (`is_climate_sensitive`).
2. Run local 5-fold cross-validation using the exact official Zindi metric:
   $$\text{Score} = 0.6 \times \text{F1-Score} + 0.4 \times \text{ROC-AUC}$$
   with decision threshold fixed at **0.5** for `TargetF1`.
3. Improve feature engineering in `src/feature_engineering.py` and models in `src/model.py`.
4. Log every experiment in `experiments.csv`.
5. Maintain the highest-scoring test predictions in `submission/best_submission.csv`.

---

## 📁 Repository Structure

```text
zindi-climate-risk/
├── .github/workflows/evaluate.yml   # CI/CD validation workflow
├── requirements.txt                 # Dependencies (pandas, scikit-learn, lightgbm, etc.)
├── experiments.csv                  # Experiment tracking log (F1, AUC, Combined score)
├── CHALLENGE_DETAILS.md             # In-depth competition specifications
├── data/
│   ├── Train.csv                    # Labeled training set (3,146 rows)
│   ├── Test.csv                     # Test set for submission (1,030 rows)
│   ├── climate_features.csv         # Enriched environmental features (CHIRPS, ERA5, MODIS)
│   ├── data_dictionary.csv          # Metadata and column descriptions
│   └── SampleSubmission.csv         # Required output format
├── notebooks/
│   └── climate_health_starter_notebook_.ipynb  # Original starter notebook
├── src/
│   ├── evaluate.py                  # Zindi metric evaluator (0.6*F1 + 0.4*AUC)
│   ├── feature_engineering.py       # Data preparation & feature transforms
│   ├── model.py                     # Model configurations (Logistic, LightGBM, XGBoost)
│   └── train.py                     # Stratified CV & submission generator
└── submission/
    └── best_submission.csv          # Output submission file for Zindi upload
```

---

## 🚀 Quickstart Commands for Jules

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Local Evaluation & Validation
```bash
# Baseline Logistic Regression (5 folds)
python src/train.py --model logistic_regression --folds 5

# LightGBM Classifier (5 folds)
python src/train.py --model lgbm --folds 5

# XGBoost Classifier (5 folds)
python src/train.py --model xgboost --folds 5
```

### 3. Submission Output
The script automatically generates and updates:
- `submission/best_submission.csv` (1,030 rows with `ID`, `TargetF1`, `TargetRAUC`).
- `experiments.csv` with CV metrics for continuous comparison.
