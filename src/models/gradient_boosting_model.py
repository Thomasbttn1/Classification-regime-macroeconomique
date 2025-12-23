"""
Gradient boosting model pour classification de régime macro
"""
from sklearn.ensemble import GradientBoostingClassifier
from base_model import BaseClassifier
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

#Setup logger
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class GradientBoostingModel(BaseClassifier):
    def __init__(self, config_path = 'config.yaml') -> None:
        super().__init__(config_path)
        self.model_name : str = "GradientBoostingModel"
        logger.info(f"{self.model_name} initialisé")

    def build_model(self) -> GradientBoostingClassifier:
        """
        Construit le modèle avec les paramètres de config.yaml
        """

        gb_config : Dict[str, Any] = self.config['models']['gradient_boosting']

        logger.info("Configuration du modèle Gradient Boosting")

        model : GradientBoostingClassifier = GradientBoostingClassifier(
            n_estimators=gb_config['n_estimators'],
            max_depth=gb_config['max_depth'],
            learning_rate= gb_config['learning_rate'],
            min_samples_split=gb_config.get('min_samples_split', 2),
            subsample=gb_config.get('subsample', 1.0), 
            random_state= gb_config['random_state'],
            verbose=0
        )

        return model
    
    def get_feature_importance(self, top_n : Optional[int]=None) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} non entrainé")
        importances = self.model.feature_importances_

        importance_df : pd.DataFrame = pd.DataFrame({
            'feature' : self.feature_names,
            'importance' : importances
        })

        #Trier
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_training_scores(self) -> Dict[str, np.ndarray]:
        """
        Retourne les scores d'entrainements à chaque itération
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} non entrainé")
        return {
            'train_scores' : self.model.train_score_
        }
    
if __name__ == "__main__":
    """
    Exécution du module du modèle Gradient Boosting
    """

    import sys
    sys.path.append('..')

    try:
        logger.info("Gradient Boosting")

        #Charger les csv
        logger.info("Chargement des données depuis data/processed/splits/...")
        splits_dir : Path = Path("data/processed/splits")

        if not splits_dir.exists():
            raise ValueError("Fichier non existant. Exécutez le fichier data_splits.py")
        
        X_train : pd.DataFrame = pd.read_csv(splits_dir / "X_train.csv", index_col=0, parse_dates=True)
        y_train : pd.Series= pd.read_csv(splits_dir / "y_train.csv", index_col=0, parse_dates=True).squeeze()

        X_val : pd.DataFrame = pd.read_csv(splits_dir / "X_val.csv", index_col=0, parse_dates=True)
        y_val : pd.Series = pd.read_csv(splits_dir / "y_val.csv", index_col=0, parse_dates=True).squeeze()

        X_test : pd.DataFrame = pd.read_csv(splits_dir / "X_test.csv", index_col=0, parse_dates=True)
        y_test : pd.Series = pd.read_csv(splits_dir / "y_test.csv", index_col=0, parse_dates=True).squeeze()

        logger.info("Fichier csv chargés")

        #Entrainement du modèle
        logger.info("Entrainement du modèle")

        model : GradientBoostingModel = GradientBoostingModel()
        metrics : Dict[str, float] = model.train(X_train, y_train, X_val, y_val)

        #Evaluation sur test
        logger.info("Evaluation sur test split")
        test_acc : float = model.score(X_test, y_test)
        logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

        #Feature importance
        importance : pd.DataFrame = model.get_feature_importance(top_n=5)
        logger.info(importance.to_string(index=False))

        #Analyse de la convergence
        logger.info("Analyse de la convergence")
        scores : Dict[str, np.ndarray] = model.get_training_scores()
        train_scores = scores['train_scores']

        logger.info(f"Score initial: {train_scores[0]:.4f}")
        logger.info(f"Score final:   {train_scores[-1]:.4f}")
        logger.info(f"Amélioration:  {train_scores[-1] - train_scores[0]:.4f}")

        # sauvegarde du modèle
        logger.info("Sauvegarde du modèle")
        model_path : str = model.save()
        logger.info(f"Modèle sauvegardé {model_path}")

        #Résume
        logger.info("Résumé")
        logger.info(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
        if 'val_accuracy' in metrics:
            logger.info(f"Val Accuracy:   {metrics['val_accuracy']:.2%}")
        logger.info(f"Test Accuracy:  {test_acc:.2%}")
        logger.info(f"Training Time:  {metrics['training_time']:.2f}s")
        logger.info(f"Top Feature:    {importance.iloc[0]['feature']}")
        logger.info(f"Top Importance: {importance.iloc[0]['importance']:.3f}")

        logger.info("Execution terminée")

    except Exception as e:
        logger.error(f"Erreur : {e}")
        import traceback
        traceback.print_exc()



