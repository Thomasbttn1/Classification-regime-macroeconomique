"""
Collecte des 4 indicateurs économiques

Indicateurs collectés : 
1. ISM Manufactuing PMI (NAPMPMI Index)
2. Consumer Sentiment (CONSSENT Index)
3. Fed Funds Rate (FDTR Index)
4. CPI US MoM (CPI CHNG Index)
"""

import pandas as pd
import numpy as np 
from fredapi import Fred
from datetime import datetime
import yaml
import logging
from pathlib import Path
import os
from typing import Dict, Optional, List, Tuple, Any, Union
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

#Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconomicDataCollector:
    """
    Collecteur de données via FRED
    """

    def __init__(self, config_path : str = 'config.yaml') :
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any]= yaml.safe_load(f)
        self.start_date : str = self.config['data']['start_date']
        self.end_date : str = self.config['data']['end_date']
        self.data_path : Path = Path(self.config['paths']['data_raw'])
        self.data_path.mkdir(parents = True, exist_ok = True)
        self.fred_api_key : Optional[str] = os.getenv('FRED_API_KEY')

        if not self.fred_api_key:
            logger.warning('Pas de clé FRED API')
            logger.warning('Créer un fichier .env avec : FRED_API_KEY=Votre clé')
        self.fred : Optional[Fred] = Fred(api_key=self.fred_api_key)

        logger.info(f"Collecteur initialisé")

    def collect_ism_pmi(self) -> pd.DataFrame:
        logger.info("Collecte de l'ISM Manufacturing PMI")

        try:
            #Telecharger depuis FRED
            data : pd.Series = self.fred.get_series(
                'ISRATIO',
                observation_start=self.start_date,
                observation_end=self.end_date
            )

            df : pd.DataFrame = pd.DataFrame({'ISM_PMI': data})

            logger.info(f"ISM PMI collecté: {len(df)} observations")

            return df
        except Exception as e:
            logger.error(f"Erreur lors de la collecte de l'ISM : {e}")
            return
    
    def collect_consumer_sentiment(self) -> pd.DataFrame:
        logger.info("Collect du Consumer Sentiment")

        try:
            data : pd.Series = self.fred.get_series(
                'UMCSENT',
                observation_start=self.start_date,
                observation_end=self.end_date
            )
            df : pd.DataFrame = pd.DataFrame({'Consumer_Sentiment' : data})

            logger.info(f"Consumer Sentiment collécté : {len(df)} observations")
            return df
        
        except Exception as e:
            logger.error(f"Erreur lors de la collecte du Consumer sentiment : {e} ")
            return
    
    def collect_fed_rate(self) -> pd.DataFrame:
        logger.info("Collecte du Fed Funds Rate")

        try:
            data : pd.Series = self.fred.get_series(
                'FEDFUNDS',
                observation_start=self.start_date,
                observation_end=self.end_date
            )
            df : pd.DataFrame = pd.DataFrame({'Fed_Rate' : data})

            logger.info(f"Fed Funds Rate collecté : {len(df)} observations")
            return df
        except Exception as e :
            logger.error(f"Erreur lors de la collecte du Fed Funds Rate : {e}")
            return
    
    def collect_cpi(self) -> pd.DataFrame:
        try:
            data : pd.Series = self.fred.get_series(
                'CPIAUCSL',
                observation_start= self.start_date,
                observation_end=self.end_date
            )
            df : pd.DataFrame = pd.DataFrame({'CPI' : data})
            
            #Calculer la variation MoM
            df['CPI_MoM'] = df['CPI'].pct_change() * 100
            df = df[['CPI_MoM']].dropna()

            logger.info(f"CPI Collecté : {len(df)} observations")
            return df
        except Exception as e :
            logger.error(f"Erreur lors de la collecte du CPI : {e}")
            return
    
    def collect_all(self) -> pd.DataFrame:
        """
        Collecte tous les indicateurs et les combine.
        """
        logger.info("Collecte des 4 indicateurs")

        ism : pd.DataFrame = self.collect_ism_pmi()
        sentiment : pd.DataFrame = self.collect_consumer_sentiment()
        fed_rate : pd.DataFrame = self.collect_fed_rate()
        cpi : pd.DataFrame = self.collect_cpi()

        data : pd.DataFrame = pd.concat([ism, sentiment, fed_rate, cpi], axis = 1)

        data = data.dropna()
        #Mettre les dates en fin de mois
        data.index = data.index.to_period('M').to_timestamp('M')
        logger.info(f"Data combinées collectées : {len(data)} observations")

        #sauvegarder 
        output_path : Path = self.data_path / 'economic_indicators.csv'
        data.to_csv(output_path)
        logger.info(f"Données sauvegardées : {output_path}")

        return data
    
def collect_economic_data() -> pd.DataFrame:
        collector : EconomicDataCollector = EconomicDataCollector()
        return collector.collect_all()

if __name__ == "__main__":
    # Test du module
    data: pd.DataFrame = collect_economic_data() 
    print("\n" + "="*70)
    print("TEST DU MODULE - COLLECTE DES DONNÉES ÉCONOMIQUES")
    print("="*70)
    print(f"\nDonnées collectées: {data.shape}")
    print(f"\nPremières lignes:")
    print(data.head())
    print(f"\nDernières lignes:")
    print(data.tail())
    