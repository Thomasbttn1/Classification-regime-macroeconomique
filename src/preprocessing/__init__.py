"""
Module de preprocessing et feature engineering

Ce module transforme les données brutes en features prêtes pour le ML:
- Feature engineering : Créer les 14 features à partir des 4 indicateurs
- Normalisation : Z-score sur fenêtre glissante
- Labeling : Création des labels RESTRICTIF/NEUTRE/ACCOMMODANT
- Train/test split : Split temporel pour éviter le data leakage
"""

__all__ = [
    'create_features',
    'normalize_features',
    'label_regimes',
    'temporal_train_test_split'
]