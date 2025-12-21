"""
Collecte les données du S&P500
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import yaml
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    Collecteur des données de marché via Yahoo Finance
    """

    def __init__(self, config_path : str = 'config.yaml') -> None:
        with open(config_path, 'r') as f:
            self.config : Dict[str, Any] = yaml.safe_load(f)
        
        self.start_date : str = self.config['data']['start_date']
        self.end_date : str = self.config['data']['end_date']
        self.spx_ticker : str = self.config['data']['tickers']['spx']
        self.data_path : Path = Path(self.config['paths']['data_raw'])
        self.data_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Collecteur de données de marché initialisé")
    
    def collect_spx(self) -> pd.DataFrame:
        logger.info("Collecte des données du SP500")

        try:
            # Telechargement via yfinance
            spx: pd.DataFrame = yf.download(
                self.spx_ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            # Gardons uniquement le close price 
            data : pd.DataFrame = pd.DataFrame({'SPX_Close' : spx[('Close', '^GSPC')]})

            logger.info(f"Données du SP500 collectées : {len(data)} observations")
            return data
        
        except Exception as e :
            logger.error(f"Erreur lors de la collecte des données du SP500 : {e}" )
            return 
    
    def calculate_forward_returns(self, data : pd.DataFrame, windows : List[int] = [20, 60, 120]) -> pd.DataFrame:
        """
        Fonction qui calcule les rendements futurs sur différentes fenêtres pour le futur labelling
        
        Exemple :
        Date          SPX_Close   SPX_Return_Forward_60d
        2020-01-31    3,225       -19.9%  ← (2584/3225-1)*100
        2020-02-29    2,954       -25.6%  ← Crash COVID
        2020-03-31    2,584        35.8%  ← Forte reprise
        2020-04-30    2,912        14.2%  ← Rebond
        """
        logger.info("Calcule des rendements futurs ")

        result : pd.DataFrame = data.copy()

        for window in windows:
            col_name : str = f'SPX_Return_Forward_{window}d'
            result[col_name] = (result['SPX_Close'].shift(-window)/result['SPX_Close']-1) * 100 #En pourcentage
        
        logger.info('Rendements forward calculés')

        return result
    
    def resample_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Rééchantillonne les données journalières en données mensuelles
        """
        logger.info("Rééchantillonnage des données en fréquence mensuelle")

        #On prend la dernière valeur de chaque fois
        monthly: pd.DataFrame = data.resample('M').last()

        logger.info(f"Données mensuelles : {len(monthly)} observations")

        return monthly
    
    def collect_all(self) -> pd.DataFrame :
        logger.info("Collecte des données de marchés")
        
        #Recupérer les données
        spx_daily: pd.DataFrame = self.collect_spx()
        #Calculer les returns
        spx_with_return : pd.DataFrame = self.calculate_forward_returns(spx_daily)
        #Transformer en données monthly
        spx_monthly : pd.DataFrame = self.resample_to_monthly(spx_with_return)

        logger.info(f"Données de marché prêtes : {len(spx_monthly)} observations")

        #Sauvegarder
        output_path : Path = self.data_path / 'market_data.csv'
        spx_monthly.to_csv(output_path)
        logger.info("Données sauvegardées")

        return spx_monthly
    
def collect_market_data() -> pd.DataFrame:
    collector : MarketDataCollector = MarketDataCollector()
    return collector.collect_all()

if __name__ == '__main__':
    #Test du module
    data : pd.DataFrame = collect_market_data()
    print("\n" + "="*70)
    print("TEST DU MODULE - COLLECTE DES DONNÉES DE MARCHÉ")
    print("="*70)
    print(f"\nDonnées collectées: {data.shape}")
    print(f"Colonnes: {list(data.columns)}")
    print(f"\nPremières lignes:")
    print(data.head())
    print(f"\nDernières lignes:")
    print(data.tail())
    print(f"\nStatistiques des rendements 60j:")
    print(data['SPX_Return_Forward_60d'].describe())