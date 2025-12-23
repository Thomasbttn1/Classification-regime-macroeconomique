"""
Classe abstraite qui définit l'interface commune pour les différents modèles. Cela garantit que les modèles soient interchangeables,
ils ont les mêmes méthodes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    def __init__(self, config_path : str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any] = yaml.safe_load(f)
        # Attributs du modèle
        self.model : Optional[Any] = None
        self.model_name : str = self.__class__.__name__
        self.is_trained : bool = False

        #Métadonnées
        self.training_date : Optional[str] = None
        self.training_time : Optional[float] = None
        self.feature_names : Optional[list] = None
        self.classes : Optional[np.ndarray] = None

        logger.info("Classe initialisé")

    @abstractmethod
    def build_model(self) -> Any:
        """
        Construit le modèles avec les paramètres de la config
        Abstract -> cette méthode doit être implémentée par les classes enfants.
        """
        pass

    def train(self, X_train: pd.DataFrame, y_train : pd.DataFrame, X_val : Optional[pd.DataFrame] = None, y_val : Optional[pd.DataFrame] = None
              ) -> Dict[str, float] :
        """
        Entraine le modèle sur les données train
        """
        logger.info("Entrainement du modèle")

        #Gérer les NaN
        n_nan = X_train.isna().sum().sum()
        if n_nan>0:
            logger.warning(f"Il y a {n_nan} NaN")
            logger.info("Remplissage des NaN avec la médiane")
            X_train = X_train.fillna(X_train.median())
            if X_val is not None:
                X_val = X_val.fillna(X_val.median())
            logger.info("NaN remplis")
        
        #Construite le modele
        if self.model is None:
            logger.info("Construction du modèle")
            self.model = self.build_model()
            logger.info("Construction terminée")
        
        #Sauvegarder les noms des features et des classes
        self.feature_names = list(X_train.columns)
        logger.info(f"\nEntraînement sur {len(X_train)} samples...")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Classes: {y_train.unique().tolist()}")
        
        import time
        start_time = time.time()
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train.values if hasattr(y_train, 'values') else y_train)
        
        # Calculer le temps
        self.training_time = time.time() - start_time
        self.training_date = datetime.now().isoformat()
        self.is_trained = True
        self.classes = self.model.classes_
        
        logger.info(f"✓ Entraînement terminé en {self.training_time:.2f}s")

        #Evaluer train et val
        train_accuracy = self.model.score(X_train, y_train)
        logger.info(f"✓ Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

        metrics : Dict[str, Any] = {
            'train_accuracy' : train_accuracy,
            'training_time' : self.training_time
        }
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            logger.info(f"✓ Val Accuracy:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            metrics['val_accuracy'] = val_accuracy

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur des nouvelles données.
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} non entrainé ")
        
        return self.model.predict(X)
    
    def predict_proba(self, X : pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} non entrainé ")
        
        return self.model.predict_proba(X)
    
    def score(self, X : pd.DataFrame, y : pd.Series) -> float:
        """
        Calcule l'accuracy sur l'ensemble des données.
        """
        if not self.is_trained:
            raise ValueError(
                f"{self.model_name} n'est pas entraîné. "
                "Appelez train() d'abord."
            )
        
        return self.model.score(X, y)
    
    def save(self, output_dir : str = 'models/saved_models') -> str:
        """
        Sauvegarde le modèle entrainé et ses métadonnées
        """
        if not self.is_trained:
            raise ValueError(
                f"Impossible de sauvegarder {self.model_name}: "
                "le modèle n'est pas entraîné."
            )

        #Créer le repo
        output_path : Path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        #nom du ficher
        model_filename = f"{self.model_name.lower()}.pkl"
        model_path = output_path / model_filename

        #Sauvegarde du modèle
        logger.info(f"Sauvegarde du modèle : {model_path}" )
        joblib.dump(self.model, model_path)

        #Sauvegarde des métadonnées
        metadata = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'training_date': self.training_date,
            'training_time': self.training_time,
            'features_name': self.feature_names,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'n_features': len(self.feature_names) if self.feature_names else None
        }

        metadata_path = output_path / f"{self.model_name.lower()}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modèle sauvegardé: {model_path}")
        logger.info(f"Métadonnées sauvegardées: {metadata_path}")
        
        return str(model_path)
    
    def load(self, model_path : str) -> None:
        """
        Charge un modèle sauvegardé
        """
        logger.info(f"Chargement du modèle {model_path}")

        #Charger le modèle
        self.model = joblib.load(model_path)
        self.is_trained = True

        #Charger les métadonnées
        metadata_path : Path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_date = metadata.get('training_date')
                self.training_time = metadata.get('training_time')
                self.feature_names = metadata.get('feature_names')
                self.classes = np.array(metadata['classes']) if metadata.get('classes') else None
        logger.info(f"✓ Modèle chargé: {type(self.model).__name__}")
        logger.info(f"✓ Entraîné le: {self.training_date}")

    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres du modèle.
        """
        if self.model is None:
            return {}
        return self.model.get_params()
    def __repr__(self) -> str:
        """
        Representation string du modèle
        """
        status = 'entrainé' if self.is_trained else 'non entrainé'
        return f"{self.model_name} (status = {status})"
    
if __name__ == '__main__':
    # Test de la class
    logger.info("La classe BasrClassifier est abstraite")
    logger.info("Elle ne peut pas être instanciée directement.")
    logger.info("Utilisez RandomForestModel, GradientBoostingModel ou MLPModel.")

    