# 🌍 Climate Risk and Health Prediction Challenge

> **Plateforme** : [Zindi](https://zindi.world/competitions/climate-risk-health-prediction-challenge)
> **Prix** : $1 000 USD
> **Période** : 18 août 2026 – 18 octobre 2026

---

## 📋 Description du Challenge

L'objectif est de construire un modèle de machine learning qui prédit si un décès enregistré
appartient à une **catégorie climatiquement sensible** (`is_climate_sensitive`).

Les données combinent des informations démographiques (zone, genre, âge, localisation)
avec des variables climatiques et environnementales enrichies (températures, précipitations,
NDVI, élévation, pente).

---

## 📊 Métrique d'Évaluation

$$\text{Score} = 0.6 \times \text{F1-Score} + 0.4 \times \text{ROC-AUC}$$

- **`TargetF1`** : prédiction binaire (0 ou 1) avec seuil fixe à **0.5**
- **`TargetRAUC`** : probabilité continue (utilisée pour le ROC-AUC)

---

## 📁 Fichiers de Données

| Fichier | Taille | Description |
|---------|--------|-------------|
| `Train.csv` | 478.5 KB | Données d'entraînement (3 146 lignes, 13 colonnes) |
| `Test.csv` | 138.5 KB | Données de test (1 030 lignes, 12 colonnes) |
| `climate_features.csv` | 1 MB | Features climatiques enrichies (4 176 lignes, 18 colonnes) |
| `data_dictionary.csv` | 749 B | Dictionnaire des colonnes principales |
| `downloaded_climate_features_data_dictionary.csv` | 2.1 KB | Dictionnaire des features climatiques |
| `SampleSubmission.csv` | 16.1 KB | Format de soumission attendu |

### Colonnes principales (Train/Test)
- `ID` — Identifiant unique
- `zone` — Rural / Peri_urban
- `gender` — Male / Female
- `deathdate` — Date du décès
- `age` — Âge au moment du décès
- `avg_temperature`, `max_temperature`, `min_temperature` — Températures
- `precipitation` — Précipitations
- `latitude`, `longitude` — Coordonnées géographiques
- `location` — Nom du lieu (39 valeurs uniques)
- `is_climate_sensitive` — **Cible binaire** (Train uniquement)

### Features climatiques enrichies
- **Pluie** : `rain_sum_7d`, `rain_sum_30d`, `rain_sum_90d`, `rain_days_30d`, `max_daily_rain_30d`
- **Température** : `tavg_7d`, `tavg_30d`, `tavg_90d`, `tmax_30d`, `tmin_30d`, `hot_days_30d`, `temp_range_mean_30d`
- **Végétation** : `ndvi_30d`, `ndvi_90d`
- **Terrain** : `elevation`, `slope`

---

## 📤 Format de Soumission

Le fichier CSV de soumission doit contenir exactement 3 colonnes :

```csv
ID,TargetF1,TargetRAUC
ID_E760D84B,0,0.32
ID_6EDEA907,1,0.87
...
```

- **`TargetF1`** : 0 ou 1 (seuil à 0.5)
- **`TargetRAUC`** : probabilité entre 0 et 1

---

## 🏗️ Structure du Dépôt

```text
zindi-climate-risk/
├── .github/workflows/evaluate.yml    # CI/CD — validation automatique
├── .gitignore
├── requirements.txt
├── CHALLENGE_DETAILS.md              # Ce fichier
├── experiments.csv                   # Journal des expériences
├── notebooks/
│   └── climate_health_starter_notebook_.ipynb
├── data/
│   ├── Train.csv
│   ├── Test.csv
│   ├── climate_features.csv
│   ├── data_dictionary.csv
│   └── downloaded_climate_features_data_dictionary.csv
├── src/
│   ├── evaluate.py                   # Métrique Zindi exacte
│   ├── feature_engineering.py        # Pipeline de features
│   ├── model.py                      # Configuration du modèle
│   └── train.py                      # Script principal d'entraînement + CV
└── submission/
    └── best_submission.csv           # Meilleure soumission courante
```

---

## 🚀 Utilisation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'entraînement + validation croisée
python src/train.py

# 3. Le fichier de soumission sera généré dans submission/best_submission.csv
```

---

## 📊 Baseline (Starter Notebook)

Le starter notebook utilise un **Logistic Regression** avec :
- OneHotEncoding pour les features catégorielles (`zone`, `gender`)
- StandardScaler + Imputation médiane pour les features numériques
- Exclusion de `location`, `latitude`, `longitude`

**Résultat baseline** : ~0.69 (F1) / ~0.72 (AUC) → Score combiné ~0.70

---

## 🎯 Pistes d'Amélioration

1. **Modèles avancés** : LightGBM, XGBoost, CatBoost, ensemble/stacking
2. **Feature engineering** :
   - Interactions température × précipitation
   - Catégories d'âge (nourrisson, enfant, adulte, senior)
   - Encoding cyclique pour `day_of_year`
   - Features lag/rolling sur séries temporelles climatiques
   - Target encoding pour `location`
3. **Validation** : StratifiedKFold (5–10 folds) au lieu d'un simple train/test split
4. **Optimisation** : Threshold tuning pour F1 (optimiser le seuil indépendamment)
5. **Sélection de features** : Permutation importance, SHAP
