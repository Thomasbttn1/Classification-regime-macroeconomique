"""
Prédiction du Régime pour 2026

Ce script utilise le modèle Random Forest entraîné pour prédire
le régime macroéconomique attendu pour l'année 2026.

Le régime prédit guidera l'allocation d'actifs stratégique.

Usage:
    python src/allocation/predict_regime_2026.py
"""

from pathlib import Path
from typing import Dict
import logging

import pandas as pd
import numpy as np
import sys

# Ajouter le chemin vers src/models
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from random_forest_model import RandomForestModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimePredictor2026:
    """Prédit le régime macroéconomique pour 2026."""
    
    def __init__(self):
        """Initialise le prédicteur."""
        self.model = None
        logger.info("Prédicteur de régime initialisé")
    
    def load_model(self, model_path: str = 'models/saved_models/randomforestmodel.pkl'):
        """
        Charge le modèle Random Forest entraîné.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        logger.info(f"\Chargement du modèle...")
        logger.info(f"  {model_path}")
        
        self.model = RandomForestModel()
        self.model.load(model_path)
        
        logger.info("Modèle chargé")
    
    def load_latest_features(
        self,
        splits_dir: str = 'data/processed/splits'
    ) -> pd.DataFrame:
        """
        Charge les features les plus récentes (fin 2025).
        
        On utilise le test set qui contient les données les plus récentes.
        
        Args:
            splits_dir: Répertoire des splits
        
        Returns:
            DataFrame avec dernières features
        """
        logger.info(f"Chargement des features récentes...")
        
        splits_path = Path(splits_dir)
        
        # Charger X_test (données les plus récentes)
        X_test = pd.read_csv(splits_path / 'X_test.csv', index_col=0, parse_dates=True)
        
        logger.info(f"Features chargées")
        logger.info(f"  Période: {X_test.index[0].date()} → {X_test.index[-1].date()}")
        logger.info(f"  Dernière date: {X_test.index[-1].date()}")
        
        return X_test
    
    def predict_regime(self, X: pd.DataFrame) -> Dict[str, any]:
        """
        Prédit le régime pour 2026 basé sur les dernières données.
        
        Args:
            X: Features (dernières observations)
        
        Returns:
            Dictionnaire avec prédiction et probabilités
        """
        logger.info(f"Prédiction du régime pour 2026...")
        
        # Prendre les 3 derniers mois pour une prédiction robuste
        recent_data = X.tail(3)
        
        # Nettoyer NaN
        recent_data_clean = recent_data.fillna(recent_data.median())
        
        # Prédire
        regimes = self.model.predict(recent_data_clean)
        probas = self.model.predict_proba(recent_data_clean)
        
        # Prendre la prédiction la plus récente
        latest_regime = regimes[-1]
        latest_probas = probas[-1]
        
        # Mapper les probabilités
        regime_classes = ['ACCOMMODANT', 'NEUTRE', 'RESTRICTIF']
        proba_dict = {
            regime_classes[i]: latest_probas[i] 
            for i in range(len(regime_classes))
        }
        
        # Tendance sur les 3 derniers mois
        regime_counts = pd.Series(regimes).value_counts()
        
        result = {
            'regime_predit': latest_regime,
            'probabilites': proba_dict,
            'tendance_3mois': regime_counts.to_dict(),
            'date_derniere_observation': X.index[-1]
        }
        
        return result
    
    def display_prediction(self, result: Dict[str, any]) -> None:
        """
        Affiche les résultats de prédiction.
        
        Args:
            result: Dictionnaire de résultats
        """
        logger.info("PRÉDICTION POUR 2026")
        
        logger.info(f"Basé sur les données jusqu'à: {result['date_derniere_observation'].date()}")
        
        logger.info(f"RÉGIME PRÉDIT: {result['regime_predit']}")
        
        logger.info(f"PROBABILITÉS:")
        for regime, proba in sorted(result['probabilites'].items(), 
                                    key=lambda x: x[1], reverse=True):
            bar_length = int(proba * 50)
            bar = '█' * bar_length
            logger.info(f"  {regime:12s}: {proba:5.1%} {bar}")
        
        logger.info(f"TENDANCE SUR LES 3 DERNIERS MOIS:")
        for regime, count in result['tendance_3mois'].items():
            logger.info(f"  {regime:12s}: {count} mois")
        
        # Interprétation
        logger.info(f"INTERPRÉTATION:")
        regime = result['regime_predit']
        proba = result['probabilites'][regime]
        
        if regime == 'ACCOMMODANT':
            logger.info(f"Conditions favorables attendues pour 2026")
            logger.info(f"     → Croissance économique soutenue")
            logger.info(f"     → Environnement propice aux actifs risqués")
            logger.info(f"     → Recommandation: Surpondérer les actions")
        elif regime == 'NEUTRE':
            logger.info(f"Conditions normales attendues pour 2026")
            logger.info(f"     → Croissance modérée")
            logger.info(f"     → Environnement équilibré")
            logger.info(f"     → Recommandation: Allocation équilibrée actions/obligations")
        else:  # RESTRICTIF
            logger.info(f"Conditions restrictives attendues pour 2026")
            logger.info(f"     → Ralentissement économique probable")
            logger.info(f"     → Environnement défensif")
            logger.info(f"     → Recommandation: Privilégier les obligations et la qualité")
        
        if proba < 0.5:
            logger.info(f"ATTENTION: Probabilité modérée ({proba:.1%})")
            logger.info(f"  Le régime pourrait évoluer. Surveillance nécessaire.")
    
    def save_prediction(
        self,
        result: Dict[str, any],
        output_path: str = 'results/tables/regime_prediction_2026.csv'
    ) -> None:
        """
        Sauvegarde la prédiction.
        
        Args:
            result: Résultats de prédiction
            output_path: Chemin de sortie
        """
        logger.info(f"Sauvegarde de la prédiction...")
        
        # Créer DataFrame
        prediction_df = pd.DataFrame([{
            'date_prediction': pd.Timestamp.now(),
            'date_derniere_observation': result['date_derniere_observation'],
            'regime_predit_2026': result['regime_predit'],
            'proba_ACCOMMODANT': result['probabilites']['ACCOMMODANT'],
            'proba_NEUTRE': result['probabilites']['NEUTRE'],
            'proba_RESTRICTIF': result['probabilites']['RESTRICTIF']
        }])
        
        # Sauvegarder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(output_path, index=False)
        
        logger.info(f"Prédiction sauvegardée: {output_path}")
    
    def run(self) -> Dict[str, any]:
        """
        Exécute la prédiction complète.
        
        Returns:
            Résultats de prédiction
        """
        logger.info("PRÉDICTION DU RÉGIME MACROÉCONOMIQUE POUR 2026")
        
        # 1. Charger modèle
        self.load_model()
        
        # 2. Charger features
        X = self.load_latest_features()
        
        # 3. Prédire
        result = self.predict_regime(X)
        
        # 4. Afficher
        self.display_prediction(result)
        
        # 5. Sauvegarder
        self.save_prediction(result)
        
        logger.info("PRÉDICTION TERMINÉE")
        
        return result


if __name__ == "__main__":
    """Exécution de la prédiction."""
    
    try:
        predictor = RegimePredictor2026()
        result = predictor.run()
        
        logger.info("Prédiction prête pour le rapport BLOC 3 !")
        logger.info("À inclure dans le rapport:")
        logger.info(f"  'Notre modèle prédit un régime {result['regime_predit']} pour 2026")
        logger.info(f"   avec une probabilité de {result['probabilites'][result['regime_predit']]:.0%}.'")
        
    except Exception as e:
        logger.error(f" Erreur: {e}")
        import traceback
        traceback.print_exc()