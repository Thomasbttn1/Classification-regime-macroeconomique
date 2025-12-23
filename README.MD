# Classification des Régimes Macroéconomiques pour l'Allocation d'Actifs 2026

**Projet Master 272 - Gestion d'Actifs**  
Université Paris Dauphine-PSL  
Décembre 2025

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Objectifs du projet](#objectifs-du-projet)
3. [Méthodologie](#méthodologie)
4. [Structure du projet](#structure-du-projet)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [Résultats](#résultats)
8. [Technologies utilisées](#technologies-utilisées)

---

## 🎯 Vue d'ensemble

Ce projet développe une stratégie d'allocation d'actifs quantitative basée sur la classification des régimes macroéconomiques. En combinant analyse fondamentale, Machine Learning et backtesting historique, nous proposons une allocation optimisée pour 2026 sur 8 classes d'actifs (actions US/Europe, obligations américaines/européennes, marché monétaire).

**Approche en 3 étapes :**
1. **Outil quantitatif** : Financial Conditions Index (FCI) pour identifier le régime économique
2. **Prévision ML** : Modèle Random Forest pour prédire le régime futur
3. **Allocation adaptative** : Règles d'allocation optimisées par régime

---

## 🎓 Objectifs du projet

### Objectif principal
Proposer une allocation stratégique d'actifs pour 2026 basée sur une classification rigoureuse des régimes macroéconomiques.

### Objectifs spécifiques
- Développer un indicateur composite (FCI) synthétisant 4 variables macro clés
- Entraîner un modèle ML pour classifier les régimes (RESTRICTIF/NEUTRE/ACCOMMODANT)
- Valider l'approche par backtesting historique (2010-2025)
- Optimiser le couple rendement-risque (Sharpe ratio)
- Générer une allocation concrète pour 2026

---

## 🔬 Méthodologie

### 1. Financial Conditions Index (FCI)

**Formule :**
```
FCI = 0,25 × ISM_norm + 0,25 × Sentiment_norm - 0,25 × Fed_norm - 0,25 × CPI_norm
```

**4 Indicateurs :**
- ISM Manufacturing PMI : Activité industrielle
- Consumer Sentiment Index : Confiance des ménages
- Fed Funds Rate : Politique monétaire
- CPI Month-over-Month : Inflation

**Normalisation :** Z-scores sur fenêtre glissante de 12 mois

**Classification :**
- RESTRICTIF : FCI < -0,5
- NEUTRE : -0,5 ≤ FCI ≤ 0,5
- ACCOMMODANT : FCI > 0,5

### 2. Machine Learning

**Modèle retenu :** Random Forest Classifier
- Features : 14 variables (indicateurs + moyennes mobiles + z-scores + volatilités)
- Split temporel : 80% train / 20% test (pas de shuffle pour préserver temporalité)
- Validation : Matrice de confusion, feature importance, métriques par classe

**Performance :**
- Accuracy : 55,17%
- Sharpe ratio backtesting : 0,50 (vs 0,16 benchmark)

### 3. Allocation d'Actifs

**8 Classes d'actifs :**
1. Actions US (SPX Index)
2. Actions EURO (SX5E Index)
3. Taux longs US 10Y (USGG10YR Index)
4. Taux courts US 2Y (USGG2YR Index)
5. Taux longs Allemands (GDBR10 Index)
6. Taux courts Allemands (GDBR2 Index)
7. Taux longs Français (GFRN10 Index)
8. Marché monétaire EURO (ESTRON Index)

**Règles d'allocation par régime :**
- RESTRICTIF : 35% actions / 65% obligations (défensif)
- NEUTRE : 70% actions / 30% obligations (équilibré)
- ACCOMMODANT : 75% actions / 25% obligations (opportuniste modéré)

---

## 📁 Structure du projet

```
classification-regimes-macro/
│
├── README.md                          # Ce fichier
├── config.yaml                        # Configuration du projet
├── requirements.txt                   # Dépendances Python
│
├── data/
│   ├── raw/                          # Données brutes
│   │   ├── economic_indicators.csv   # ISM, Sentiment, Fed, CPI (2010-2025)
│   │   ├── multi_asset_returns.csv   # Rendements mensuels 8 actifs
│   │   └── multi_asset_prices.csv    # Prix historiques 8 actifs
│   │
│   └── processed/                    # Données traitées
│       ├── features.csv              # Features ML (188 obs × 14 features)
│       ├── labeled_data.csv          # Données labellées avec régimes
│       └── splits/                   # Train/test splits
│
├── src/
│   ├── data_collection/              # Collecte de données
│   │   ├── collect_economic_data.py  # FRED API (ISM, Sentiment, Fed, CPI)
│   │   └── collect_multi_assets.py   # Yahoo Finance (8 actifs)
│   │
│   ├── preprocessing/                # Préparation des données
│   │   ├── feature_engineering.py    # Création features ML
│   │   ├── labeling.py               # Labeling régimes via FCI
│   │   └── data_split.py             # Split temporel train/test
│   │
│   ├── indicators/                   # Indicateurs économiques
│   │   └── financial_conditions_index.py  # Calcul FCI + graphiques
│   │
│   ├── models/                       # Modèles ML
│   │   ├── base_model.py             # Classe abstraite
│   │   ├── random_forest_model.py    # Random Forest (modèle retenu)
│   │   ├── model_evaluator.py        # Évaluation + confusion matrix
│   │   └── backtest_strategy.py      # Backtesting allocation
│   │
│   └── allocation/                   # Génération allocation 2026
│       ├── predict_regime_2026.py    # Prédiction régime 2026
│       └── generate_allocation_2026.py  # Allocation finale
│
├── models/
│   └── saved_models/                 # Modèles entraînés
│       └── randomforestmodel.pkl     # Modèle Random Forest
│
└── results/
    ├── figures/                      # Graphiques pour rapport
    │   ├── fci_evolution.png         # Évolution FCI (ANNEXE 1)
    │   ├── confusion_matrices.png    # Matrices de confusion
    │   ├── feature_importance.png    # Importance des features
    │   ├── backtest_results.png      # Performance backtesting
    │   └── allocation_2026.png       # Allocation 2026
    │
    └── tables/                       # Tableaux CSV
        ├── fci_timeseries.csv        # Série temporelle FCI
        ├── regime_prediction_2026.csv  # Prédiction 2026
        ├── allocation_2026.csv       # Allocation finale
        └── backtest_metrics.csv      # Métriques backtesting
```

---

## 🛠️ Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/classification-regimes-macro.git
cd classification-regimes-macro
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer l'API FRED (obligatoire pour collecter les données économiques)**

a) Obtenir une clé API gratuite :
   - Aller sur https://fred.stlouisfed.org/
   - Créer un compte (gratuit)
   - Demander une API Key : https://fredaccount.stlouisfed.org/apikeys

b) Créer un fichier `.env` à la racine du projet :
```bash
touch .env  # Linux/Mac
# ou créer manuellement sur Windows
```

c) Ajouter votre clé API dans `.env` :
```
FRED_API_KEY=votre_cle_api_ici
```

Exemple :
```
FRED_API_KEY=abc123def456ghi789jkl0mn
```

d) **IMPORTANT** : Ajouter `.env` au `.gitignore` pour ne pas versionner votre clé :
```bash
echo ".env" >> .gitignore
```

**Note** : Le fichier `.env` ne doit JAMAIS être partagé ou commité sur Git (clé privée).

---

## 🚀 Utilisation

### Pipeline complet

#### 1. Collecte des données
```bash
# Données économiques (FRED API)
python src/data_collection/collect_economic_data.py

# Données marchés (Yahoo Finance)
python src/data_collection/collect_multi_assets.py
```

#### 2. Préparation des données
```bash
# Feature engineering
python src/preprocessing/feature_engineering.py

# Labeling via FCI
python src/preprocessing/labeling.py

# Split train/test
python src/preprocessing/data_split.py
```

#### 3. Entraînement du modèle
```bash
# Entraîner Random Forest
python src/models/random_forest_model.py

# Évaluer
python src/models/model_evaluator.py
```

#### 4. Backtesting
```bash
python src/models/backtest_strategy.py
```

#### 5. Génération allocation 2026
```bash
# FCI actuel
python src/indicators/financial_conditions_index.py

# Prédire régime 2026
python src/allocation/predict_regime_2026.py

# Générer allocation
python src/allocation/generate_allocation_2026.py
```

---

## 📊 Résultats

### État actuel (septembre 2025)
- **FCI** : -0,38 → NEUTRE
- **ISM PMI** : 1,4
- **Consumer Sentiment** : 55,1
- **Fed Funds Rate** : 4,22%
- **CPI MoM** : 0,31%

### Prédiction 2026
- **Régime prédit** : ACCOMMODANT (48,8%)
- **Interprétation** : Conditions favorables, croissance 2-2,5%

### Allocation 2026

| Actif | Allocation |
|-------|------------|
| Actions US | 52% |
| Actions EURO | 23% |
| Taux longs US | 10% |
| Taux courts US | 5% |
| Taux Allemands | 8% |
| Taux Français | 1% |
| Cash EURO | 1% |

**Total Actions : 75% / Obligations : 25%**

### Performance Backtesting (2010-2025)

| Métrique | Stratégie | Benchmark | SPX |
|----------|-----------|-----------|-----|
| Rendement ann. | **8,03%** | 2,93% | 12,38% |
| Volatilité | 12,13% | 5,84% | 14,45% |
| **Sharpe** | **0,50** | 0,16 | 0,72 |
| Drawdown | -21,93% | -7,75% | -24,77% |

✅ **Sharpe 3x supérieur au benchmark**

---

## 🧰 Technologies utilisées

- **Python 3.8+**
- **pandas, numpy** : Manipulation données
- **scikit-learn** : Machine Learning
- **matplotlib, seaborn** : Visualisations
- **yfinance** : Données marchés
- **fredapi** : Données économiques

---

## 📝 Améliorations futures

- [ ] Plus d'indicateurs (VIX, spreads crédit)
- [ ] Autres algos ML (XGBoost, LSTM)
- [ ] Rebalancing dynamique
- [ ] Interface web (Streamlit)
- [ ] Tests unitaires

---

## 👤 Auteur

**Thomas Betton & Pierre Berthold**  
Master 272 - Université Paris Dauphine-PSL  
Décembre 2025

---

