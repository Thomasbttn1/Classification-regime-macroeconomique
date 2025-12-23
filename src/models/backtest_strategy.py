"""
Module de Backtesting - Allocation d'Actifs Basée sur les Régimes

Ce module utilise les prédictions de régimes macroéconomiques pour définir
une stratégie d'allocation d'actifs et la backtest sur données historiques.

Stratégie d'allocation:
    - RESTRICTIF:   20% actions, 80% obligations (défensif)
    - NEUTRE:       60% actions, 40% obligations (équilibré)
    - ACCOMMODANT:  80% actions, 20% obligations (agressif)

Métriques calculées:
    - Rendement total
    - Rendement annualisé
    - Volatilité annualisée
    - Sharpe ratio
    - Maximum drawdown
    - Taux de réussite

Usage:
    python src/backtesting/backtest_strategy.py
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from random_forest_model import RandomForestModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

# Style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class AssetAllocationBacktester:
    """
    Backtester pour stratégie d'allocation basée sur les régimes.
    
    Example:
        >>> backtester = AssetAllocationBacktester()
        >>> results = backtester.run_backtest()
        >>> backtester.plot_results(results)
    """
    
    def __init__(
        self,
        allocation_rules: Dict[str, Dict[str, float]] = None,
        output_dir: str = 'results/backtesting'
    ) -> None:
        """
        Initialise le backtester.
        
        Args:
            allocation_rules: Règles d'allocation par régime
            output_dir: Répertoire pour sauvegarder les résultats
        """
        # Règles d'allocation AGRESSIVES (optimisées pour Sharpe élevé)
        # Adaptées à la distribution réelle : 51% NEUTRE / 47% ACCOMMODANT / 2% RESTRICTIF
        if allocation_rules is None:
            self.allocation_rules = {
                'RESTRICTIF': {
                    # Défensif mais garde des actions (rare : 2% du temps)
                    'SPX': 0.20,      # 20% actions US
                    'SX5E': 0.15,     # 15% actions Euro
                    'USGG10YR': 0.20, # Taux longs US 10Y
                    'USGG2YR': 0.12,  # Taux courts US 2Y
                    'GDBR10': 0.15,   # Taux longs Allemands
                    'GDBR2': 0.08,    # Taux courts Allemands
                    'GFRN10': 0.06,   # Taux longs Français
                    'ESTRON': 0.04    # Marché monétaire Euro
                },
                'NEUTRE': {
                    # AGRESSIF car 51% du temps ! Maximiser exposition actions
                    'SPX': 0.50,      # 50% actions US
                    'SX5E': 0.20,     # 20% actions Euro
                    'USGG10YR': 0.10, # Taux longs US 10Y
                    'USGG2YR': 0.06,  # Taux courts US 2Y
                    'GDBR10': 0.06,   # Taux longs Allemands
                    'GDBR2': 0.04,    # Taux courts Allemands
                    'GFRN10': 0.02,   # Taux longs Français
                    'ESTRON': 0.02    # Marché monétaire Euro
                },
                'ACCOMMODANT': {
                    # Opportuniste modéré : 75% actions (47% du temps)
                    'SPX': 0.52,      # 52% actions US
                    'SX5E': 0.23,     # 23% actions Euro
                    'USGG10YR': 0.10, # Taux longs US 10Y
                    'USGG2YR': 0.05,  # Taux courts US 2Y
                    'GDBR10': 0.05,   # Taux longs Allemands
                    'GDBR2': 0.03,    # Taux courts Allemands
                    'GFRN10': 0.01,   # Taux longs Français
                    'ESTRON': 0.01    # Marché monétaire Euro
                }
            }
        else:
            self.allocation_rules = allocation_rules
        
        # Vérifier que les allocations somment à 100%
        for regime, alloc in self.allocation_rules.items():
            total = sum(alloc.values())
            if not np.isclose(total, 1.0):
                logger.warning(f"{regime}: allocation totale = {total:.1%} (devrait être 100%)")
        
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Backtester initialisé")
        logger.info(f"  Stratégie d'allocation AGRESSIVE sur 8 classes d'actifs:")
        logger.info(f"  (Optimisée pour maximiser le Sharpe ratio)")
        
        # Afficher les allocations
        asset_names = {
            'SPX': 'Actions US',
            'SX5E': 'Actions Euro',
            'USGG10YR': 'Taux longs US 10Y',
            'USGG2YR': 'Taux courts US 2Y',
            'GDBR10': 'Taux longs Allemands',
            'GDBR2': 'Taux courts Allemands',
            'GFRN10': 'Taux longs Français',
            'ESTRON': 'Marché monétaire Euro'
        }
        
        for regime in ['RESTRICTIF', 'NEUTRE', 'ACCOMMODANT']:
            alloc = self.allocation_rules[regime]
            total_equity = alloc['SPX'] + alloc['SX5E']
            total_bonds = sum(v for k, v in alloc.items() if k not in ['SPX', 'SX5E'])
            
            logger.info(f"\n  {regime:12s} ({total_equity:.0%} actions / {total_bonds:.0%} obligations):")
            for asset, weight in alloc.items():
                if weight >= 0.05:  # Afficher seulement si >= 5%
                    logger.info(f"    {asset_names[asset]:30s}: {weight:5.1%}")
    
    def load_model(self, model_path: str = 'models/saved_models/randomforestmodel.pkl'):
        """Charge le modèle Random Forest sauvegardé."""
        logger.info(f"Chargement du modèle...")
        logger.info(f"  {model_path}")
        
        self.model = RandomForestModel()
        self.model.load(model_path)
        
        logger.info("Modèle chargé")
    
    def load_market_data(
        self,
        returns_file: str = 'data/raw/multi_asset_returns.csv'
    ) -> pd.DataFrame:
        """
        Charge les rendements des 8 actifs depuis le fichier local.
        
        Args:
            returns_file: Chemin vers multi_asset_returns.csv
        
        Returns:
            DataFrame avec rendements mensuels des 8 actifs
        """
        logger.info(f"Chargement des données de marché...")
        logger.info(f"  Fichier: {returns_file}")
        
        # Charger les rendements
        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        
        # Vérifier les colonnes disponibles
        expected_cols = ['SPX_return', 'SX5E_return', 'USGG10YR_return', 'USGG2YR_return',
                        'GDBR10_return', 'GDBR2_return', 'GFRN10_return', 'ESTRON_return']
        
        available = [col for col in expected_cols if col in returns_df.columns]
        missing = [col for col in expected_cols if col not in returns_df.columns]
        
        if missing:
            logger.warning(f"Colonnes manquantes: {missing}")
        
        logger.info(f"Données chargées: {len(returns_df)} mois")
        logger.info(f"  Période: {returns_df.index[0].date()} → {returns_df.index[-1].date()}")
        logger.info(f"  Actifs: {len(available)}")
        
        return returns_df
    
    def load_features_data(
        self,
        splits_dir: str = 'data/processed/splits'
    ) -> pd.DataFrame:
        """
        Charge les features pour faire des prédictions.
        
        Returns:
            DataFrame complet avec toutes les features
        """
        logger.info(f"Chargement des features...")
        
        splits_path = Path(splits_dir)
        
        # Charger les 3 splits
        X_train = pd.read_csv(splits_path / 'X_train.csv', index_col=0, parse_dates=True)
        X_val = pd.read_csv(splits_path / 'X_val.csv', index_col=0, parse_dates=True)
        X_test = pd.read_csv(splits_path / 'X_test.csv', index_col=0, parse_dates=True)
        
        # Concaténer
        X_full = pd.concat([X_train, X_val, X_test]).sort_index()
        
        logger.info(f"Features chargées: {len(X_full)} mois")
        
        return X_full
    
    def predict_regimes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les régimes sur les données historiques.
        
        Args:
            X: Features
        
        Returns:
            DataFrame avec régimes et probabilités
        """
        logger.info(f"Prédiction des régimes...")
        
        # Nettoyer NaN
        X_clean = X.fillna(X.median())
        
        # Prédire
        regimes = self.model.predict(X_clean)
        probas = self.model.predict_proba(X_clean)
        
        # Créer DataFrame
        regime_df = pd.DataFrame({
            'regime': regimes,
            'proba_ACCOMMODANT': probas[:, 0],
            'proba_NEUTRE': probas[:, 1],
            'proba_RESTRICTIF': probas[:, 2]
        }, index=X.index)
        
        logger.info(f"Régimes prédits")
        logger.info(f"  Distribution:")
        for regime, count in regime_df['regime'].value_counts().items():
            pct = count / len(regime_df) * 100
            logger.info(f"    {regime:12s}: {count:3d} mois ({pct:5.1f}%)")
        
        return regime_df
    
    def calculate_portfolio_returns(
        self,
        regimes: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcule les rendements du portefeuille selon la stratégie sur 8 actifs.
        
        Args:
            regimes: Prédictions de régimes
            market_data: Rendements des 8 actifs
        
        Returns:
            DataFrame avec rendements du portefeuille
        """
        logger.info(f"Calcul des rendements du portefeuille...")
        
        # Aligner les données
        combined = regimes.join(market_data, how='inner')
        
        # Mapper les noms de colonnes
        asset_cols = {
            'SPX': 'SPX_return',
            'SX5E': 'SX5E_return',
            'USGG10YR': 'USGG10YR_return',
            'USGG2YR': 'USGG2YR_return',
            'GDBR10': 'GDBR10_return',
            'GDBR2': 'GDBR2_return',
            'GFRN10': 'GFRN10_return',
            'ESTRON': 'ESTRON_return'
        }
        
        # Vérifier les colonnes disponibles
        available_assets = {k: v for k, v in asset_cols.items() if v in combined.columns}
        
        if len(available_assets) < len(asset_cols):
            missing = set(asset_cols.keys()) - set(available_assets.keys())
            logger.warning(f"Actifs manquants: {missing}")
        
        # Calculer le rendement du portefeuille pour chaque mois
        portfolio_returns = []
        
        for idx, row in combined.iterrows():
            regime = row['regime']
            allocation = self.allocation_rules[regime]
            
            # Rendement = Somme(poids_i × rendement_i)
            monthly_return = 0.0
            for asset_code, weight in allocation.items():
                col_name = asset_cols.get(asset_code)
                if col_name and col_name in combined.columns:
                    asset_return = row[col_name]
                    if pd.notna(asset_return):
                        monthly_return += weight * asset_return
            
            portfolio_returns.append(monthly_return)
        
        combined['portfolio_return'] = portfolio_returns
        
        # Benchmark : allocation équipondérée fixe (12.5% chacun)
        benchmark_returns = []
        equal_weight = 1.0 / len(available_assets)
        
        for idx, row in combined.iterrows():
            monthly_return = 0.0
            for col_name in available_assets.values():
                if col_name in combined.columns:
                    asset_return = row[col_name]
                    if pd.notna(asset_return):
                        monthly_return += equal_weight * asset_return
            benchmark_returns.append(monthly_return)
        
        combined['benchmark_return'] = benchmark_returns
        
        # Benchmark actions : 100% SPX
        if 'SPX_return' in combined.columns:
            combined['stocks_return'] = combined['SPX_return']
        
        # Performance cumulée
        combined['portfolio_cumulative'] = (1 + combined['portfolio_return']).cumprod()
        combined['benchmark_cumulative'] = (1 + combined['benchmark_return']).cumprod()
        if 'stocks_return' in combined.columns:
            combined['stocks_cumulative'] = (1 + combined['stocks_return']).cumprod()
        
        logger.info(f"Rendements calculés sur {len(combined)} mois")
        
        return combined
    
    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcule les métriques de performance.
        
        Args:
            returns_df: DataFrame avec rendements
        
        Returns:
            Dictionnaire de métriques
        """
        logger.info(f"Calcul des métriques de performance...")
        
        metrics = {}
        
        for strategy in ['portfolio', 'benchmark', 'stocks']:
            returns = returns_df[f'{strategy}_return']
            cumulative = returns_df[f'{strategy}_cumulative']
            
            # Rendement total
            total_return = cumulative.iloc[-1] - 1
            
            # Rendement annualisé
            n_years = len(returns) / 12
            annual_return = (1 + total_return) ** (1 / n_years) - 1
            
            # Volatilité annualisée
            annual_vol = returns.std() * np.sqrt(12)
            
            # Sharpe ratio (assume risk-free = 2%)
            sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
            
            # Maximum drawdown
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            metrics[strategy] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'final_value': cumulative.iloc[-1]
            }
        
        logger.info("Métriques calculées")
        
        return metrics
    
    def plot_results(
        self,
        returns_df: pd.DataFrame,
        metrics: Dict[str, Dict[str, float]]
    ) -> None:
        """Crée les visualisations des résultats."""
        logger.info(f"Création des visualisations...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Performance cumulée
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(returns_df.index, returns_df['portfolio_cumulative'], 
                label='Stratégie Régimes', linewidth=2, color='#2E86AB')
        ax1.plot(returns_df.index, returns_df['benchmark_cumulative'], 
                label='Benchmark 60/40', linewidth=2, color='#A23B72', linestyle='--')
        ax1.plot(returns_df.index, returns_df['stocks_cumulative'], 
                label='S&P 500 (100%)', linewidth=1.5, color='#F18F01', alpha=0.7)
        ax1.set_title('Performance Cumulée des Stratégies', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Valeur du Portfolio ($1 initial)')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        running_max = returns_df['portfolio_cumulative'].expanding().max()
        drawdown = (returns_df['portfolio_cumulative'] - running_max) / running_max
        ax2.fill_between(returns_df.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax2.plot(returns_df.index, drawdown * 100, color='darkred', linewidth=1.5)
        ax2.set_title('Drawdown de la Stratégie Régimes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des régimes
        ax3 = fig.add_subplot(gs[1, 1])
        regime_counts = returns_df['regime'].value_counts()
        colors = {'RESTRICTIF': '#E63946', 'NEUTRE': '#F4A261', 'ACCOMMODANT': '#2A9D8F'}
        ax3.bar(regime_counts.index, regime_counts.values, 
               color=[colors[r] for r in regime_counts.index])
        ax3.set_title('Distribution des Régimes Prédits', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Nombre de Mois')
        for i, v in enumerate(regime_counts.values):
            ax3.text(i, v + 1, str(v), ha='center', fontweight='bold')
        
        # 4. Allocation moyenne par régime
        ax4 = fig.add_subplot(gs[2, 0])
        
        regimes = ['RESTRICTIF', 'NEUTRE', 'ACCOMMODANT']
        assets = ['SPX', 'SX5E', 'USGG10YR', 'USGG2YR', 'GDBR10', 'GDBR2', 'GFRN10', 'ESTRON']
        colors_assets = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#F2CC8F', '#81B29A']
        
        # Créer un stacked bar chart
        bottom = np.zeros(len(regimes))
        
        for i, asset in enumerate(assets):
            values = [self.allocation_rules[r][asset] * 100 for r in regimes]
            ax4.bar(regimes, values, bottom=bottom, label=asset, color=colors_assets[i])
            bottom += values
        
        ax4.set_ylabel('Allocation (%)')
        ax4.set_title('Allocation par Régime (8 Classes d\'Actifs)', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Comparaison des métriques
        ax5 = fig.add_subplot(gs[2, 1])
        metrics_names = ['Rendement\nAnnualisé', 'Volatilité\nAnnualisée', 'Sharpe\nRatio', 'Max\nDrawdown']
        strategy_data = [
            metrics['portfolio']['annual_return'] * 100,
            metrics['portfolio']['annual_volatility'] * 100,
            metrics['portfolio']['sharpe_ratio'],
            metrics['portfolio']['max_drawdown'] * 100
        ]
        benchmark_data = [
            metrics['benchmark']['annual_return'] * 100,
            metrics['benchmark']['annual_volatility'] * 100,
            metrics['benchmark']['sharpe_ratio'],
            metrics['benchmark']['max_drawdown'] * 100
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        ax5.bar(x - width/2, strategy_data, width, label='Stratégie Régimes', color='#2E86AB')
        ax5.bar(x + width/2, benchmark_data, width, label='Benchmark 60/40', color='#A23B72')
        ax5.set_ylabel('Valeur')
        ax5.set_title('Comparaison des Métriques', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_names, fontsize=9)
        ax5.legend()
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Backtesting de la Stratégie d\'Allocation Basée sur les Régimes', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Sauvegarder
        output_path = self.output_dir / 'backtest_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphiques sauvegardés: {output_path}")
        plt.close()
    
    def save_metrics_table(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Sauvegarde les métriques dans un tableau CSV."""
        logger.info(f"Sauvegarde des métriques...")
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'Stratégie'
        
        # Formater en pourcentages
        for col in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown']:
            metrics_df[col] = metrics_df[col] * 100
        
        metrics_df = metrics_df.round(2)
        
        output_path = self.output_dir / 'backtest_metrics.csv'
        metrics_df.to_csv(output_path)
        logger.info(f"Métriques sauvegardées: {output_path}")
        
        return metrics_df
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Exécute le backtest complet.
        
        Returns:
            Dictionnaire avec tous les résultats
        """
        logger.info("BACKTESTING DE LA STRATÉGIE D'ALLOCATION")
        
        # 1. Charger le modèle
        self.load_model()
        
        # 2. Charger les données
        market_data = self.load_market_data()
        features = self.load_features_data()
        
        # 3. Prédire les régimes
        regimes = self.predict_regimes(features)
        
        # 4. Calculer les rendements
        returns_df = self.calculate_portfolio_returns(regimes, market_data)
        
        # 5. Calculer les métriques
        metrics = self.calculate_metrics(returns_df)
        
        # 6. Visualiser
        self.plot_results(returns_df, metrics)
        
        # 7. Sauvegarder
        metrics_df = self.save_metrics_table(metrics)
        
        # Afficher les résultats
        logger.info("RÉSULTATS DU BACKTESTING")
        logger.info(f"\n{metrics_df.to_string()}")
        
        return {
            'returns_df': returns_df,
            'metrics': metrics,
            'metrics_df': metrics_df,
            'regimes': regimes
        }


if __name__ == "__main__":
    """Exécution du backtesting."""
    
    try:
        # Créer le backtester
        backtester = AssetAllocationBacktester()
        
        # Lancer le backtest
        results = backtester.run_backtest()
        
        logger.info("BACKTESTING TERMINÉ")
        logger.info("Fichiers générés:")
        logger.info("  • results/backtesting/backtest_results.png")
        logger.info("  • results/backtesting/backtest_metrics.csv")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()