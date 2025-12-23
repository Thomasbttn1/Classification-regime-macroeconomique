"""
Module pour spliter les données en Train/Val/Test
Contrairement à un split aléatoire, le split temporel respecte l'ordre chronologique:
- TRAIN : données les plus anciennes (ex: 2010-2017) 70%
- VALIDATION : données intermédiaires (ex: 2017-2019) 15%
- TEST : données les plus récentes (ex: 2019-2025) 15%

Ceci est crucial pour les séries temporelles car :
1. On ne peut pas prédire le passé avec le futur
2. Simule un cas d'usage réel (prédire le futur avec les données passées)
3. Évite les fuites d'information (data leakage)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class TemporalDataSplitter:
    """
    Class pour split les données en Train/Val/Test.
    """
    def __init__(self, config_path : str = 'config.yaml') -> None:
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any] = yaml.safe_load(f)
        # Paramètres de split
        self.train_ratio : float = self.config['split']['train_ratio']
        self.val_ratio : float = self.config['split']['val_ratio']
        self.test_ratio : float = self.config['split']['test_ratio']
        self.method : str = self.config['split']['method']
        self.random_state : int = self.config['models']['random_forest']['random_state']

        # Définir les colonnes de features 
        self.features_columns : List[str] = [
            'ISM_PMI', 'Consumer_Sentiment', 'Fed_Rate', 'CPI_MoM',
            'ISM_change_3m', 'Sentiment_change_3m', 'Fed_change_12m', 'CPI_Annualized',
            'ISM_MA_6m', 'Sentiment_MA_6m', 'Fed_MA_6m', 'CPI_MA_6m',
            'ISM_Zscore_12m', 'Sentiment_Zscore_12m'
        ]

        self.target_column : str = 'Regime'

        #Initialiser le Scaler 
        self.scaler : StandardScaler = StandardScaler()
        
        logger.info('Temporal Data Splitter initialisé')
    
    def temporal_split(self, data : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
        """
        Découpe les donneés : renvoie un tuple avec les splits.
        """
        logger.info("Split des données temporel")

        #Vérifier que les données sont dans l'ordre chronologique
        if not data.index.is_monotonic_increasing:
            logger.warning("Les données ne sont pas triés par ordre chronologique -> Tri automatique")
            data = data.sort_index()
        
        #Calculer les indices de split
        n_samples : int = len(data)
        train_end_idx : int = int(n_samples * self.train_ratio)
        val_end_idx : int = int(n_samples * (self.val_ratio + self.train_ratio))

        #Découpage des données 
        train_data : pd.DataFrame = data.iloc[:train_end_idx].copy()
        val_data : pd.DataFrame = data[train_end_idx:val_end_idx].copy()
        test_data : pd.DataFrame = data[val_end_idx:].copy()

        logger.info('Split terminé')
        return train_data, val_data, test_data
    
    def prepare_features_and_labels(self, data : pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sépare les features X et les labels y.
        Returns:
        Tuple (X, y) où :
        - X : DataFrame des features (14 colonnes)
        - y : Series des labels (RESTRICTIF/NEUTRE/ACCOMMODANT)
        """
        #Vérifier que toutes les features existent
        missing_features : List[str] = [col for col in self.features_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f'Missing columns : {missing_features}')
        #Vérifier que le target existe
        if self.target_column not in data.columns:
            raise ValueError("Missing target")
        #Séparer X et y
        X : pd.DataFrame = data[self.features_columns].copy()
        y : pd.Series = data[self.target_column].copy()
        
        return X, y
    
    def normalize_features(
            self, X_train : pd.DataFrame, X_val : pd.DataFrame, X_test : pd.DataFrame
            )-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """
        Normalise les features avec standardScaler. Le scaler est fit uniquement sur les données d'entrainement pour éviter le leakage.
        """
        logger.info("Normalisation des features")
        #Fit sur train uniquement
        self.scaler.fit(X_train)

        #Transform sur val et test
        X_train_scaled : np.ndarray = self.scaler.transform(X_train)
        X_val_scaled : np.ndarray = self.scaler.transform(X_val)
        X_test_scaled : np.ndarray = self.scaler.transform(X_test)

        # Convertir en dataframe
        X_train_scaled_df : pd.DataFrame = pd.DataFrame(
            X_train_scaled,
            index=X_train.index,
            columns=X_train.columns
        )
        X_val_scaled_df : pd.DataFrame = pd.DataFrame(
            X_val_scaled,
            index=X_val.index,
            columns=X_val.columns
        )

        X_test_scaled_df : pd.DataFrame = pd.DataFrame(
            X_test_scaled,
            index=X_test.index,
            columns=X_test.columns
        )

        logger.info("Normalisation terminée")

        return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df
    
    def split_and_prepare(self, data : pd.DataFrame, normalize : bool = True) -> Dict[str, Any]:
        """
        Pipeline complet : split + séparation (features / label) + normalisation
        """
        #Split 
        train_data, val_data, test_data = self.temporal_split(data)

        #Séparation features / label
        X_train, y_train = self.prepare_features_and_labels(train_data)
        X_val, y_val = self.prepare_features_and_labels(val_data)
        X_test, y_test = self.prepare_features_and_labels(test_data)

        # Normalisation
        scaler : Optional[StandardScaler] = None
        if normalize :
            X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
            scaler = self.scaler
        else:
            logger.info('Normalisation desactivée')
        
        logger.info('Pipeline terminé')

        return {
            #Features
            'X_train' : X_train,
            'X_val' : X_val,
            'X_test' : X_test,

            #Labels
            'y_train' : y_train,
            'y_val' : y_val,
            'y_test' : y_test,

            #Métadonnées
            'scaler' : scaler,
            'features_names' : self.features_columns,

            #Stats
            'n_train' : len(X_train),
            'n_val' : len(X_val),
            'n_test' : len(X_test),
            'n_features' : len(self.features_columns)
        }
    def save_splits(self,splits: Dict[str, Any],output_dir: str = 'data/processed/splits') -> None:
        """
        Sauvegarde les datasets splitées et le scaler.
        """
        logger.info("SAUVEGARDE DES SPLITS")

        # Créer le répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        #Sauvegarder les features (X)
        logger.info("\n1. Sauvegarde des features (X)...")
        for dataset_name in ['train', 'val', 'test']:
            X = splits[f'X_{dataset_name}']
            filepath = output_path / f'X_{dataset_name}.csv'
            X.to_csv(filepath)
            logger.info(f"{filepath} ({X.shape})")
        
        # Sauvegarder les labels (y)
        logger.info("2. Sauvegarde des labels (y)...")
        for dataset_name in ['train', 'val', 'test']:
            y = splits[f'y_{dataset_name}']
            filepath = output_path / f'y_{dataset_name}.csv'
            y.to_csv(filepath, header=True)
            logger.info(f"{filepath} ({len(y)} samples)")
        
        # auvegarder le scaler
        if splits['scaler'] is not None:
            logger.info("3. Sauvegarde du StandardScaler...")
            scaler_path = output_path / 'scaler.pkl'
            joblib.dump(splits['scaler'], scaler_path)
            logger.info(f"{scaler_path}")
        else:
            logger.info("3. Pas de scaler à sauvegarder (normalize=False)")
        
        # Sauvegarder les métadonnées
        logger.info("4. Sauvegarde des métadonnées...")
        
        # Extraire les informations 
        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_val = splits['y_val']
        y_test = splits['y_test']
        
        metadata = {
            'split_date': datetime.now().isoformat(),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': len(X_train.columns),
            'feature_names': list(X_train.columns),
            'normalized': splits.get('scaler') is not None,
            'train_period': {
                'start': str(X_train.index.min()),
                'end': str(X_train.index.max())
            },
            'val_period': {
                'start': str(X_val.index.min()),
                'end': str(X_val.index.max())
            },
            'test_period': {
                'start': str(X_test.index.min()),
                'end': str(X_test.index.max())
            },
            'class_distribution': {
                'train': {
                    'RESTRICTIF': int((y_train == 'RESTRICTIF').sum()),
                    'NEUTRE': int((y_train == 'NEUTRE').sum()),
                    'ACCOMMODANT': int((y_train == 'ACCOMMODANT').sum())
                },
                'val': {
                    'RESTRICTIF': int((y_val == 'RESTRICTIF').sum()),
                    'NEUTRE': int((y_val == 'NEUTRE').sum()),
                    'ACCOMMODANT': int((y_val == 'ACCOMMODANT').sum())
                },
                'test': {
                    'RESTRICTIF': int((y_test == 'RESTRICTIF').sum()),
                    'NEUTRE': int((y_test == 'NEUTRE').sum()),
                    'ACCOMMODANT': int((y_test == 'ACCOMMODANT').sum())
                }
            }
        }
        
        metadata_path = output_path / 'split_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"{metadata_path}")
        

        logger.info("SAUVEGARDE TERMINÉE")

        logger.info(f"Tous les fichiers sont dans: {output_path.absolute()}")
        logger.info("Fichiers créés:")
        logger.info("X_train.csv, X_val.csv, X_test.csv")
        logger.info("y_train.csv, y_val.csv, y_test.csv")
        logger.info("scaler.pkl (StandardScaler)")
        logger.info("split_info.json (métadonnées)")


    def load_splits(self,input_dir: str = 'data/processed/splits') -> Dict[str, Any]:
        """
        Charge les datasets splitées depuis les fichiers sauvegardés.
        """
        logger.info("CHARGEMENT DES SPLITS SAUVEGARDÉS")
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(
                f"Le répertoire {input_path} n'existe pas.\n"
                "Assurez-vous d'avoir exécuté save_splits() d'abord."
            )
        
        # Charger les features (X)
        logger.info("Chargement des features (X)...")
        X_train = pd.read_csv(input_path / 'X_train.csv', index_col=0, parse_dates=True)
        X_val = pd.read_csv(input_path / 'X_val.csv', index_col=0, parse_dates=True)
        X_test = pd.read_csv(input_path / 'X_test.csv', index_col=0, parse_dates=True)
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"X_val: {X_val.shape}")
        logger.info(f"X_test: {X_test.shape}")
        
        # Charger les labels (y)
        logger.info("Chargement des labels (y)...")
        y_train = pd.read_csv(input_path / 'y_train.csv', index_col=0, parse_dates=True).squeeze()
        y_val = pd.read_csv(input_path / 'y_val.csv', index_col=0, parse_dates=True).squeeze()
        y_test = pd.read_csv(input_path / 'y_test.csv', index_col=0, parse_dates=True).squeeze()
        logger.info(f"y_train: {len(y_train)} samples")
        logger.info(f"y_val: {len(y_val)} samples")
        logger.info(f"y_test: {len(y_test)} samples")
        
        # Charger le scaler
        scaler_path = input_path / 'scaler.pkl'
        if scaler_path.exists():
            logger.info("Chargement du StandardScaler...")
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler chargé")
        else:
            logger.info("3. Pas de scaler sauvegardé")
            scaler = None
        
        # Charger les métadonnées
        logger.info("4. Chargement des métadonnées...")
        metadata_path = input_path / 'split_info.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Split créé le: {metadata['split_date']}")
        logger.info(f"Ratios: {metadata['train_ratio']}/{metadata['val_ratio']}/{metadata['test_ratio']}")
        
        logger.info("CHARGEMENT TERMINÉ")
        
        # Retourner le même format que split_and_prepare()
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': metadata['feature_names'],
            'n_train': metadata['n_train'],
            'n_val': metadata['n_val'],
            'n_test': metadata['n_test'],
            'n_features': metadata['n_features'],
            'metadata': metadata
        }


def temporal_train_test_split(data : pd.DataFrame, normalize : bool = True) -> Dict[str, Any]:
    splitter : TemporalDataSplitter = TemporalDataSplitter()
    return splitter.split_and_prepare(data, normalize=True)

if __name__ == '__main__':
    try:
        logger.info('Test du module de split')

        #Charger les données labelisées
        data_path : Path = Path('data/processed/labeled_data.csv')

        if not data_path.exists():
            raise FileNotFoundError('Fichier non trouvé')
        logger.info("Chargement des données labelisées")

        data : pd.DataFrame = pd.read_csv(data_path, index_col=0, parse_dates=True)

        logger.info("Données chargées")

        #Faire le split
        logger.info("Split des données en cours")
        splits: Dict[str, Any] = temporal_train_test_split(data, normalize=True)

        # Sauvegarder les splits
        logger.info("\nSauvegarde des splits...")
        splitter: TemporalDataSplitter = TemporalDataSplitter()
        splitter.save_splits(splits)
        
        # Test du chargement
        logger.info("Test du chargement des splits...")
        loaded_splits = splitter.load_splits()
        logger.info("Splits rechargés avec succès")

        logger.info(" TEST TERMINÉ AVEC SUCCÈS!")

    except Exception as e:
        logger.error(f"Erreur : {e}")





