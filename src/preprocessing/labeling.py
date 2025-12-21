"""
Labelling des régimes macro. 
Crée des labels RESTRICIF / NEUTRE / ACCOMMODANT basés sur les rendements futurs du SP500. Cette approche utilise le marché comme 
proxy pour valider des conditions macro.

Logique :
- Rendement 60j > +5%  → ACCOMMODANT (conditions favorables)
- Rendement 60j < -5%  → RESTRICTIF (conditions défavorables)
- Entre -5% et +5%     → NEUTRE
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class RegimeLabeler:
    def __init__(self, config_path : str = 'config.yaml') -> None:
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any] = yaml.safe_load(f)

        self.labelling_method : str = self.config['labeling']['method']
        self.forward_window : int = self.config['labeling']['forward_window']
        self.threshold_restrictive : float = self.config['labeling']['thresholds']['restrictive']
        self.threshold_accommodating : float = self.config['labeling']['thresholds']['accommodating']

        logger.info('Regime labeller initialisé')
    
    def label_with_forward_returns(self, features : pd.DataFrame, market_data : pd.DataFrame) -> pd.DataFrame:
        """
        Labelling sur les forward returns du SP500
        """
        logger.info("Labelling avec les forward returns")
        
        #Vérifier que la colonne existe
        return_col : str = f'SPX_Return_Forward_{self.forward_window}d'
        if return_col not in market_data.columns:
            raise ValueError('Missing column')
        
        #Merge des données 
        result : pd.DataFrame = features.copy()
        #result = result.join(market_data[return_col], how='left')
        
        # APRÈS
        # Normaliser les index
        result.index = pd.to_datetime(result.index).normalize()
        market_data.index = pd.to_datetime(market_data.index).normalize()

        # Join
        result = result.join(market_data[return_col], how='left')

        # Debug
        print(f"\nAprès join, NaN: {result[return_col].isna().sum()} / {len(result)}")

        #Créer les labels selon les seuils
        conditions : List[pd.Series] = [
            result[return_col] > self.threshold_accommodating,
            result[return_col] < self.threshold_restrictive,
        ]

        choices : List[str] = ['ACCOMMODANT', 'RESTRICTIF']

        result['Regime'] = np.select(conditions, choices, default='NEUTRE')

        logger.info('Labelling terminé')

        return result
    
    def label_with_composite_score(self, features : pd.DataFrame) -> pd.DataFrame:
        """
        Labelling basé sur un score composite des features. Alternative au forward return.
        """
        logger.info("Labelling avec composite score")

        result : pd.DataFrame = features.copy()

        #Score composite = moyenne des z-scores 
        score_cols : List[str] = ['ISM_Zscore_12m', 'Sentiment_Zscore_12m']
        missing_cols : List[str] = [col for col in score_cols if col not in result.columns]

        if missing_cols :
            raise ValueError('Missing columns')
        
        #Calculer le score compisite 
        result['Compisite_Score'] = result[score_cols].mean(axis=1)

        #Labelling basé sur le score
        conditions : List[pd.Series] = [
            result['Compisite_Score'] > 1.0,
            result['Compisite_Score'] < 1.0,
        ]
        choices : List[str] = [
            'ACCOMMODANT',
            'RESTRICTIF'
        ]

        result['Regime'] = np.select(conditions, choices, default='NEUTRE')

        logger.info('Labelling avec composite score terminé')
        return result
    
    def create_label(self, features : pd.DataFrame, market_data : Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.labelling_method == 'forward_returns':
            if market_data is None :
                raise ValueError('Market data is None')
            return self.label_with_forward_returns(features, market_data)
        elif self.labelling_method == 'composite_score':
            return self.label_with_composite_score(features)
        else:
            raise ValueError('Select a correct labelling method')
        
    def analyse_regime_distribution(self, labeled_data : pd.DataFrame) -> pd.DataFrame:
        """
        Analyse la distribution des régimes dans le temps
        """
        logger.info('Analyse de la distribution des régimes')

        stats_list : List[Dict[str, Any]] = []

        for regime in ['RESTRICTIF', 'NEUTRE', 'ACCOMMODANT']:
            mask : pd.Series = labeled_data['Regime'] == regime
            count : int = mask.sum()

            stats : Dict[str, Any] = {
                'Regime' : regime,
                'Count' : count
            }
            stats_list.append(stats)
        
        stats_df : pd.DataFrame = pd.DataFrame(stats_list)
        logger.info('Statistiques par regime')
        logger.info(f"{stats_df.to_string(index = False)}")

        return stats_df

def label_regimes(features : pd.DataFrame, market_data : Optional[pd.DataFrame] = None) -> pd.DataFrame:
    labeler : RegimeLabeler = RegimeLabeler()
    return labeler.create_label(features, market_data)


if __name__ == "__main__":
    # Test du module avec VRAIES DONNÉE
    try:
        # Charger les vraies données
        logger.info("\nChargement des données...")
        
        eco_data_path = Path('data/raw/economic_indicators.csv')
        market_data_path = Path('data/raw/market_data.csv')
        
        if not eco_data_path.exists():
            raise FileNotFoundError(f"Fichier manquant: {eco_data_path}")
        if not market_data_path.exists():
            raise FileNotFoundError(f"Fichier manquant: {market_data_path}")
        
        # Charger les données économiques
        eco_data: pd.DataFrame = pd.read_csv(eco_data_path, index_col=0, parse_dates=True)
        logger.info(f"✓ Données économiques chargées: {eco_data.shape}")
        
        # Charger les données de marché
        market_data: pd.DataFrame = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
        logger.info(f"✓ Données de marché chargées: {market_data.shape}")
        
        
        features_path = Path('data/processed/features.csv')
     
        logger.info(f"\nChargement des features existantes depuis {features_path}...")
        features: pd.DataFrame = pd.read_csv(features_path, index_col=0, parse_dates=True)
        logger.info(f"✓ Features chargées: {features.shape}")
       
        # Labeling avec vraies données
        logger.info("\nLabeling avec vraies données...")
        labeled_real: pd.DataFrame = label_regimes(features, market_data)
        
        output_path = Path('data/processed/labeled_data.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_real.to_csv(output_path)
        logger.info(f"✓ Données labelées sauvegardées: {output_path}")
        
        logger.info("\n✓ Test avec vraies données terminé avec succès!")
        logger.info(f"✓ Fichier sauvegardé: data/processed/labeled_data.csv")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ Erreur: {e}")
        logger.error("Assurez-vous que les fichiers suivants existent:")
        logger.error("  - data/raw/economic_indicators.csv")
        logger.error("  - data/raw/market_data.csv")
        logger.info("\nVous pouvez les générer avec:")
        logger.info("  python src/data_collection/collect_economic_data.py")
        logger.info("  python src/data_collection/collect_market_data.py")
    except Exception as e:
        logger.error(f"\n✗ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()