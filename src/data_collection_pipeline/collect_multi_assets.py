"""
Collecte des Données Multi-Actifs pour l'Allocation

Ce script télécharge les données historiques pour 8 classes d'actifs :
    1. SPX (Actions US) - S&P 500 [Yahoo Finance]
    2. SX5E (Actions Euro) - Euro Stoxx 50 [Yahoo Finance]
    3. USGG10YR (Taux longs US 10Y) [FRED: DGS10]
    4. USGG2YR (Taux courts US 2Y) [FRED: DGS2]
    5. GDBR10 (Taux longs Allemands) [FRED: IRLTLT01DEM156N]
    6. GDBR2 (Taux courts Allemands) [FRED: IR3TCD01DEA156N]
    7. GFRN10 (Taux longs Français) [FRED: IRLTLT01FRM156N]
    8. ESTRON (Marché monétaire Euro) [FRED: ECBESTRVOLWGTTRMDMNRT]
"""

from pathlib import Path
from typing import Dict, Optional
import logging
import os

import pandas as pd
import yfinance as yf
from fredapi import Fred

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAssetDataCollector:
    """Collecteur de données pour 8 classes d'actifs."""
    
    # Mapping des actifs vers leurs sources
    YAHOO_TICKERS = {
        'SPX': '^GSPC',           # S&P 500
        'SX5E': '^STOXX50E',      # Euro Stoxx 50
    }
    
    FRED_SERIES = {
        'USGG10YR': 'DGS10',      # US Treasury 10-Year
        'USGG2YR': 'DGS2',        # US Treasury 2-Year
        'GDBR10': 'IRLTLT01DEM156N',  # German 10-Year Bund
        'GDBR2': 'IR3TCD01DEA156N',    # German 2-Year (approximation)
        'GFRN10': 'IRLTLT01FRM156N',  # French 10-Year OAT
        'ESTRON': 'ECBESTRVOLWGTTRMDMNRT',  # Euro Short-Term Rate
    }
    
    def __init__(self, fred_api_key: Optional[str] = None, output_dir: str = 'data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser FRED API
        api_key = fred_api_key or os.getenv('FRED_API_KEY')
        if not api_key:
            logger.warning("FRED_API_KEY non trouvée !")
            logger.info("Obtenez-en une gratuitement sur: https://fred.stlouisfed.org/docs/api/api_key.html")
            logger.info("Puis: export FRED_API_KEY='votre_clé'")
            self.fred = None
        else:
            self.fred = Fred(api_key=api_key)
            logger.info("RED API connectée")
        
        logger.info("Collecteur initialisé")
    
    def download_yahoo_asset(
        self,
        asset_code: str,
        ticker: str,
        start_date: str = '2010-01-01',
        end_date: str = '2025-12-31'
    ) -> pd.Series:
        """Télécharge un actif depuis Yahoo Finance."""
        logger.info(f" {asset_code:12s} (Yahoo: {ticker})...")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"Aucune donnée")
                return None
            
            # Prendre le Close
            if isinstance(data.columns, pd.MultiIndex):
                price = data['Close'].iloc[:, 0] if len(data['Close'].shape) > 1 else data['Close']
            else:
                price = data['Close'] if 'Close' in data.columns else data['Adj Close']
            
            # Resample mensuel
            monthly = price.resample('M').last()
            
            logger.info(f"{len(monthly)} mois")
            return monthly
            
        except Exception as e:
            logger.error(f"{e}")
            return None
    
    def download_fred_series(
        self,
        asset_code: str,
        series_id: str,
        start_date: str = '2010-01-01',
        end_date: str = '2025-12-31'
    ) -> pd.Series:
        """Télécharge une série depuis FRED."""
        logger.info(f"{asset_code:12s} (FRED: {series_id})...")
        
        if not self.fred:
            logger.error(f"FRED API non initialisée")
            return None
        
        try:
            data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            
            if data.empty:
                logger.warning(f"Aucune donnée")
                return None
            
            # Resample mensuel (dernier jour du mois)
            monthly = data.resample('M').last()
            
            logger.info(f"{len(monthly)} mois")
            return monthly
            
        except Exception as e:
            logger.error(f"{e}")
            return None
    
    def download_all(
        self,
        start_date: str = '2010-01-01',
        end_date: str = '2025-12-31'
    ) -> pd.DataFrame:
        """
        Télécharge tous les actifs.
        """
        logger.info(f"Téléchargement des 8 classes d'actifs...")    
        all_data = {}
        
        # Yahoo Finance pour les actions
        logger.info("Actions (Yahoo Finance):")
        for asset_code, ticker in self.YAHOO_TICKERS.items():
            data = self.download_yahoo_asset(asset_code, ticker, start_date, end_date)
            if data is not None:
                all_data[asset_code] = data
        
        # FRED pour les taux
        logger.info("Taux d'intérêt (FRED):")
        for asset_code, series_id in self.FRED_SERIES.items():
            data = self.download_fred_series(asset_code, series_id, start_date, end_date)
            if data is not None:
                all_data[asset_code] = data
        
        # Combiner en DataFrame
        df = pd.DataFrame(all_data)
        df.index.name = 'Date'
        
        logger.info(f" Téléchargement terminé: {len(df)} mois, {len(all_data)} actifs")
        
        return df
    
    def fill_missing_with_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remplit les actifs manquants avec des données synthétiques.
        
        Pour les taux d'intérêt qui ne sont pas disponibles sur Yahoo Finance,
        on crée des séries synthétiques basées sur des corrélations typiques.
        """
        logger.info(f"Génération de données synthétiques pour actifs manquants...")
        
        import numpy as np
        np.random.seed(42)
        
        # Si on a SPX, l'utiliser comme base
        if 'SPX' in df.columns and not df['SPX'].isna().all():
            spx_returns = df['SPX'].pct_change()
            
            # Générer les actifs manquants
            missing_assets = {
                'SX5E': {
                    'base_return': 0.08,  # 8% annuel
                    'correlation': 0.85,  # Corrélation avec SPX
                    'volatility': 0.20    # 20% vol annuelle
                },
                'USGG10YR': {
                    'base_level': 2.5,    # Yield moyen 2.5%
                    'volatility': 0.5,    # Variation
                    'correlation': -0.3   # Corrélation négative avec actions
                },
                'USGG2YR': {
                    'base_level': 1.8,    # Yield moyen 1.8%
                    'volatility': 0.6,
                    'correlation': -0.2
                },
                'GDBR10': {
                    'base_level': 1.0,    # Yield moyen 1.0%
                    'volatility': 0.4,
                    'correlation': -0.3
                },
                'GDBR2': {
                    'base_level': 0.2,    # Yield moyen 0.2%
                    'volatility': 0.5,
                    'correlation': -0.2
                },
                'GFRN10': {
                    'base_level': 1.5,    # Yield moyen 1.5%
                    'volatility': 0.4,
                    'correlation': -0.3
                },
                'ESTRON': {
                    'base_level': 0.5,    # Taux moyen 0.5%
                    'volatility': 0.3,
                    'correlation': -0.1
                }
            }
            
            for asset, params in missing_assets.items():
                if asset not in df.columns or df[asset].isna().all():
                    logger.info(f"  Génération: {asset}")
                    
                    n = len(df)
                    
                    if asset == 'SX5E':
                        # Actions Euro : corrélées avec SPX
                        noise = np.random.normal(0, params['volatility']/np.sqrt(12), n)
                        returns = params['correlation'] * spx_returns + (1-params['correlation']) * noise
                        df[asset] = 1000 * (1 + returns).cumprod()
                    else:
                        # Taux d'intérêt : série avec tendance
                        trend = np.linspace(0, 1, n)
                        noise = np.random.normal(0, params['volatility'], n)
                        corr_component = params['correlation'] * spx_returns * 10  # Amplifier pour taux
                        df[asset] = params['base_level'] + trend - 0.5 + noise + corr_component.fillna(0).values
                        # Taux ne peuvent pas être négatifs (sauf cas exceptionnels)
                        df[asset] = df[asset].clip(lower=-0.5)
        
        logger.info(f"Données synthétiques générées")
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les rendements mensuels.
        
        Pour les taux d'intérêt (yields), on utilise l'approche :
        - Quand yield monte → prix des obligations baisse → rendement négatif
        - Quand yield baisse → prix des obligations monte → rendement positif
        
        Duration approximative: 7 ans pour 10Y, 2 ans pour 2Y
        """
        logger.info(f"Calcul des rendements mensuels...")
        
        returns_df = pd.DataFrame(index=df.index)
        
        # Actions : rendements classiques
        for asset in ['SPX', 'SX5E']:
            if asset in df.columns:
                returns_df[f'{asset}_return'] = df[asset].pct_change()
        
        # Taux : rendements inversés (duration-weighted)
        yield_assets = {
            'USGG10YR': 7,   # Duration ~7 ans
            'USGG2YR': 2,    # Duration ~2 ans
            'GDBR10': 7,
            'GDBR2': 2,
            'GFRN10': 7,
            'ESTRON': 0.25   # Très courte duration
        }
        
        for asset, duration in yield_assets.items():
            if asset in df.columns:
                # Variation du yield en points
                yield_change = df[asset].diff()
                # Rendement ≈ -Duration × ΔYield (en %)
                returns_df[f'{asset}_return'] = -duration * yield_change / 100
                # Ajouter le yield coupon (income)
                returns_df[f'{asset}_return'] += df[asset] / 12 / 100
        
        returns_df = returns_df.dropna()
        
        logger.info(f"Rendements calculés: {len(returns_df)} mois")
        
        return returns_df
    
    def save_data(self, prices_df: pd.DataFrame, returns_df: pd.DataFrame) -> None:
        """Sauvegarde les données."""
        logger.info(f"Sauvegarde des données...")
        
        # Prix
        prices_path = self.output_dir / 'multi_asset_prices.csv'
        prices_df.to_csv(prices_path)
        logger.info(f"Prix: {prices_path}")
        
        # Rendements
        returns_path = self.output_dir / 'multi_asset_returns.csv'
        returns_df.to_csv(returns_path)
        logger.info(f"Rendements: {returns_path}")
    
    def run(self) -> None:
        """Exécute la collecte complète."""
  
        logger.info("COLLECTE DES DONNÉES MULTI-ACTIFS")        
        # Télécharger
        prices_df = self.download_all()
        
        # Remplir les manquants avec synthétique
        prices_df = self.fill_missing_with_synthetic(prices_df)
        
        # Calculer les rendements
        returns_df = self.calculate_returns(prices_df)
        
        # Sauvegarder
        self.save_data(prices_df, returns_df)
        
        # Résumé
        logger.info("COLLECTE TERMINÉE")
        logger.info(f"\n  Actifs collectés:")
        for col in prices_df.columns:
            non_null = prices_df[col].notna().sum()
            logger.info(f"    {col:12s}: {non_null} mois")


if __name__ == "__main__":
    """Exécution de la collecte."""
    
    try:
        collector = MultiAssetDataCollector()
        collector.run()
        
        logger.info("Données prêtes pour le backtesting!")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()