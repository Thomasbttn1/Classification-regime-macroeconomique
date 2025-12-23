"""
Random Forest pour la classification des régimes macro

C'est un ensemble d'arbre de décision:
    - chaque arbre vote pour une class
    - Prédiction finale = vote majoritaire
    
Paramètres:
    - n_estimators : Nombre d'abres = 200
    - max_depth : Profondeur max = 10
    - min_samples_split = 20
"""

from sklearn.ensemble import RandomForestClassifier
from base_model import BaseClassifier
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class RandomForestModel(BaseClassifier):
    """
    Modèle random forest
    """
    def __init__(self, config_path = 'config.yaml') -> None:
        super().__init__(config_path)
        self.model_name = "RandomForestModel"
        logger.info(f"{self.model_name} initialisé")
    
    def build_model(self) -> RandomForestClassifier:
        """
        Construire le modèle avec les paramètres de config.yaml
        """
        rf_config : Dict[str, Any] = self.config['models']['random_forest']
        
        #Créer le modèle avec les paramètres

        model : RandomForestClassifier = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth= rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            random_state=rf_config['random_state'],
            n_jobs=1,
            verbose=0,
            warm_start=False,
            class_weight=None
        )
        return model
    
    def get_feature_importance(self, top_n : Optional[int] = None) -> pd.DataFrame:
        """
        Retourne l'importance des features i.e quelles features sont les plus importantes pour les predictions.
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} n'est pas entrainé")
        
        #Extraire les importances
        importances = self.model.feature_importances_

        importance_df : pd.DataFrame = pd.DataFrame({
            'feature' : self.feature_names,
            'importance' : importances
        }) 
        #Trier par importance décroissante
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df
    
if __name__ == "__main__":
    """
    Test du modèle Random Forest
    """
    import sys
    sys.path.append('..')

    try:
        logger.info("Test de Random Forest")
        logger.info("Chargment des données depuis le fichier data/processed/splits")

        splits_dir : Path = Path('data/processed/splits')

        if not splits_dir.exists():
            raise FileNotFoundError("Dossier non trouvé. Veuillez exécuter data_split.py")
        
        #Charger les CSV
        X_train : pd.DataFrame = pd.read_csv(splits_dir/'X_train.csv', index_col=0,parse_dates=True)

        y_train : pd.DataFrame = pd.read_csv(splits_dir/'y_train.csv', index_col=0, parse_dates=True).squeeze()

        X_val : pd.DataFrame = pd.read_csv(splits_dir/'X_val.csv', index_col=0,parse_dates=True)

        y_val : pd.DataFrame = pd.read_csv(splits_dir/'y_val.csv', index_col=0, parse_dates=True).squeeze()

        X_test : pd.DataFrame = pd.read_csv(splits_dir/'X_test.csv', index_col=0,parse_dates=True)

        y_test : pd.DataFrame = pd.read_csv(splits_dir/'y_test.csv', index_col=0, parse_dates=True).squeeze()

        logger.info("Données chargées")

        #Entrainement du modèle
        logger.info("Entrainement du modèle")
        model : RandomForestModel = RandomForestModel()
        metrics : Dict[str, float] = model.train(X_train, y_train, X_val, y_val)

        #Evaluation sur test
        logger.info("Evaluation sur test")
        test_acc : float = model.score(X_test, y_test)
        logger.info(f"Test accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

        # Feature importance
        logger.info("Feature importance")
        importance : pd.DataFrame = model.get_feature_importance(top_n = 5)
        logger.info(importance.to_string(index=False))

        # sauvegarde du modèle
        logger.info("Sauvegarde du modèle")
        model_path : str = model.save()
        logger.info(f"Modèle sauvegardé {model_path}")

        # Résumé
        logger.info("Résumé des performances")
        logger.info(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
        if 'val_accuracy' in metrics:
            logger.info(f"Val Accuracy:   {metrics['val_accuracy']:.2%}")
        logger.info(f"Test Accuracy:  {test_acc:.2%}")
        logger.info(f"Training Time:  {metrics['training_time']:.2f}s")
        logger.info(f"Top Feature:    {importance.iloc[0]['feature']}")
        logger.info(f"Top Importance: {importance.iloc[0]['importance']:.3f}")
    except Exception as e:
        logger.error(f"Erreur : {e}")
        import traceback
        traceback.print_exc()

