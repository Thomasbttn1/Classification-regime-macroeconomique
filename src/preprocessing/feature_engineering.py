"""
Feature Engineering pour le projet d'Allocation 2026

Créé les 14 features à partir des 4 indicateurs économiques de base:
1-4.   Indicateurs bruts (ISM, Sentiment, Fed, CPI)
5-8.   Variations (3 mois pour ISM/Sentiment, 12 mois pour Fed, annualisé pour CPI)
9-12.  Moyennes mobiles 6 mois
13-14. Z-scores 12 mois (ISM et Sentiment)
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union

#Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Créateur de features (métriques) pour le modèle de classification des régimes macro
    """
    def __init__(self, config_path : str = 'config.yaml') -> None:
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any] = yaml.safe_load(f)
        # Paramètre de features depuis config.yaml

        self.window_ma : int = self.config['features']['window_ma']
        self.window_zscore : int = self.config['features']['window_zscore']

        logger.info('Features Engineering initialisé')

    def calculate_changes(self, data : pd.DataFrame) :
        """
        Calcule les variations des indicateurs
        """
        logger.info("Calcule des variations des indicateurs")

        result: pd.DataFrame = data.copy()

        # Variation 3m pour ISM et Sentiment
        window_3m : int = 3
        result['ISM_change_3m'] = result['ISM_PMI'].diff(window_3m)
        result['Sentiment_change_3m'] = result['Consumer_Sentiment'].diff(window_3m)

        #Variation 1 an pour Fed Rate
        window_12m : int = 12
        result['Fed_change_12m'] = result['Fed_Rate'].diff(window_12m)

        #CPI Annualisé
        result['CPI_Annualized'] = result['CPI_MoM']*12

        logger.info("Variations calculées")

        return result
    
    def calculate_moving_average(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le moving average sur une moyenne glissante
        """
        logger.info("Calcule des moving average")

        result : pd.DataFrame = data.copy()

        result['ISM_MA_6m'] = result['ISM_PMI'].rolling(window=self.window_ma, min_periods=1).mean()

        result['Sentiment_MA_6m'] = result['Consumer_Sentiment'].rolling(window=self.window_ma, min_periods=1).mean()

        result['Fed_MA_6m'] = result['Fed_Rate'].rolling(window=self.window_ma, min_periods=1).mean()

        result['CPI_MA_6m'] = result['CPI_MoM'].rolling(window=self.window_ma, min_periods=1).mean()

        logger.info("Moving average calculés")

        return result
    
    def calculate_zscore(self, data : pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le z-score pour une fenêtre donnée. On calcule les z-scores pour l'ISM et le Consument Sentiment uniquement car ce
        sont des variables cycliques qui oscillent autours autour d'une valeur. Il n'y a pas de comportement tendanciel pour le Fed
        Rate et le CPI donc peu d'utilité d'utiliser les z-scores comme features.
        """
        logger.info("Calcule des z-scores")

        result : pd.DataFrame = data.copy()

        #Z-score ISM sur 12 mois
        ism_rolling_mean : pd.Series = result['ISM_PMI'].rolling(window=self.window_zscore, min_periods=1).mean()
        ism_rolling_std : pd.Series = result['ISM_PMI'].rolling(window=self.window_zscore, min_periods=1).std()

        result['ISM_Zscore_12m'] = (result['ISM_PMI'] - ism_rolling_mean) / ism_rolling_std

        #Z-score Sentiment sur 12 mois
        sentiment_rolling_mean : pd.Series = result['Consumer_Sentiment'].rolling(window=self.window_zscore, min_periods=1).mean()
        sentiment_rolling_std : pd.Series = result['Consumer_Sentiment'].rolling(window=self.window_zscore, min_periods=1).std()

        result['Sentiment_Zscore_12m'] = (result['Consumer_Sentiment'] - sentiment_rolling_mean) / sentiment_rolling_std

        #Remplaçons les NaN et les inf par des 0
        result['ISM_Zscore_12m'].replace([np.inf, -np.inf], 0, inplace=True)
        result['Sentiment_Zscore_12m'].replace([np.inf, -np.inf], 0, inplace=True)
        result['ISM_Zscore_12m'].fillna(0, inplace=True)
        result['Sentiment_Zscore_12m'].fillna(0, inplace=True)

        logger.info("Zscore calculés")

        return result
    
    def calculate_all_features(self, data : pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les features et les combine dans une même dataset.
        """
        logger.info("Calculs de toutes les features")

        # Indicateurs bruts -> 4 features
        result : pd.DataFrame = data.copy()

        # Ajouter les variations -> 4 features
        result = self.calculate_changes(result)

        #Ajouter les moving average -> 4 features
        result = self.calculate_moving_average(result)

        #Ajouter les zscores -> 2 features
        result = self.calculate_zscore(result)

        #14 features en tout

        #Sauvegarde de data
        data_path : Path = Path(self.config['paths']['data_processed'])
        data_path.mkdir(parents=True, exist_ok=True)
        output_path : Path = data_path / 'features.csv'
        result.to_csv(output_path)
        logger.info(f"Données sauvegardées : {output_path}")

        return result
    
    def get_features_names(self) -> List[str]:
        """
        Retourne la liste du nom des features dans l'ordre
        """

        features_name : List[str] = [
            # Indicateurs bruts (4)
            'ISM_PMI',
            'Consumer_Sentiment',
            'Fed_Rate',
            'CPI_MoM',
            # Variations (4)
            'ISM_Change_3m',
            'Sentiment_Change_3m',
            'Fed_Change_12m',
            'CPI_Annualized',
            # Moyennes mobiles (4)
            'ISM_MA_6m',
            'Sentiment_MA_6m',
            'Fed_MA_6m',
            'CPI_MA_6m',
            # Z-scores (2)
            'ISM_ZScore_12m',
            'Sentiment_ZScore_12m'
        ]
        return features_name

def create_features(economic_data : pd.DataFrame) -> pd.DataFrame:
    engineer : FeatureEngineering = FeatureEngineering()
    return engineer.calculate_all_features(economic_data)

if __name__ == "__main__":
    # Test du module avec VRAIES DONNÉES
    logger.info("FEATURE ENGINEERING")
    
    # Charger les vraies données économiques
    try:
        eco_data_path = Path('data/raw/economic_indicators.csv')
        
        if not eco_data_path.exists():
            raise FileNotFoundError(f"Fichier manquant: {eco_data_path}")
        
        logger.info(f"\nChargement des vraies données depuis {eco_data_path}...")
        real_data: pd.DataFrame = pd.read_csv(eco_data_path, index_col=0, parse_dates=True)
        
        logger.info(f"✓ Données chargées: {real_data.shape}")
        
        # Créer les features
        features: pd.DataFrame = create_features(real_data)
        
        # Afficher les résultats
      
        logger.info("\nFeature engineering terminé avec succès!")
        logger.info(f"Fichier sauvegardé: data/processed/features.csv")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ Erreur: {e}")
        logger.error("Assurez-vous que le fichier data/raw/economic_indicators.csv existe")
        logger.info("Vous pouvez le générer avec: python src/data_collection/collect_economic_data.py")
    except Exception as e:
        logger.error(f"\n✗ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()