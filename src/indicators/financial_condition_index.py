"""
Financial Conditions Index (FCI) - BLOC 1

Module pour calculer et visualiser le Financial Conditions Index (FCI),
l'outil quantitatif principal du projet d'allocation.

Le FCI combine 4 indicateurs économiques clés :
- ISM Manufacturing PMI (activité industrielle)
- Consumer Sentiment (confiance des ménages)
- Fed Funds Rate (politique monétaire)
- CPI MoM (inflation)

Formule FCI :
    FCI = 0.25 × ISM_norm + 0.25 × Sentiment_norm - 0.25 × Fed_norm - 0.25 × CPI_norm

Classification des régimes :
    - FCI < -0.5  → RESTRICTIF (conditions financières restrictives)
    - -0.5 ≤ FCI ≤ 0.5 → NEUTRE (conditions normales)
    - FCI > 0.5  → ACCOMMODANT (conditions favorables)

Usage:
    python src/indicators/financial_conditions_index.py
"""

from pathlib import Path
from typing import Dict, Tuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class FinancialConditionsIndex:
    """
    Calcule et analyse le Financial Conditions Index (FCI).
    
    Le FCI est un indicateur composite qui mesure l'état des conditions
    financières et économiques en combinant 4 variables clés.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialise le calculateur FCI.
        
        Args:
            config_path: Chemin vers config.yaml
        """
        # Charger config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fci_config = self.config['fci']
        self.window = self.fci_config['window']
        self.weights = self.fci_config['weights']
        self.thresholds = self.fci_config['thresholds']
        
        logger.info("FCI Calculator initialisé")
        logger.info(f"  Fenêtre normalisation: {self.window} mois")
        logger.info(f"  Poids: ISM={self.weights['ism']}, Sentiment={self.weights['sentiment']}")
        logger.info(f"        Fed={self.weights['fed_rate']}, CPI={self.weights['cpi']}")
        logger.info(f"  Seuils: RESTRICTIF < {self.thresholds['restrictive']}")
        logger.info(f"          ACCOMMODANT > {self.thresholds['accommodating']}")
    
    def load_data(self, data_path: str = 'data/raw/economic_indicators.csv') -> pd.DataFrame:
        """
        Charge les données économiques.
        
        Args:
            data_path: Chemin vers economic_indicators.csv
        
        Returns:
            DataFrame avec les 4 indicateurs
        """
        logger.info(f"Chargement des données...")
        logger.info(f"  {data_path}")
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Vérifier les colonnes requises
        required = ['ISM_PMI', 'Consumer_Sentiment', 'Fed_Rate', 'CPI_MoM']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        logger.info(f"Données chargées: {len(df)} observations")
        logger.info(f"  Période: {df.index[0].date()} → {df.index[-1].date()}")
        
        return df
    
    def normalize_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les indicateurs sur une fenêtre glissante.
        
        Normalisation z-score sur fenêtre de N mois :
            z = (x - mean) / std
        
        Args:
            df: DataFrame avec indicateurs bruts
        
        Returns:
            DataFrame avec indicateurs normalisés
        """
        logger.info(f"Normalisation des indicateurs...")
        logger.info(f"  Méthode: Z-score sur fenêtre glissante de {self.window} mois")
        
        df_norm = df.copy()
        
        for col in ['ISM_PMI', 'Consumer_Sentiment', 'Fed_Rate', 'CPI_MoM']:
            # Moyenne et écart-type glissants
            rolling_mean = df[col].rolling(window=self.window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=self.window, min_periods=1).std()
            
            # Z-score
            df_norm[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std
            
            # Remplacer NaN par 0
            df_norm[f'{col}_norm'] = df_norm[f'{col}_norm'].fillna(0)
        
        logger.info(f"Normalisation terminée")
        
        return df_norm
    
    def calculate_fci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le FCI.
        
        FCI = w1×ISM + w2×Sentiment + w3×Fed + w4×CPI
        
        Args:
            df: DataFrame avec indicateurs normalisés
        
        Returns:
            DataFrame avec colonne FCI
        """
        logger.info(f"Calcul du FCI...")
        
        df['FCI'] = (
            self.weights['ism'] * df['ISM_PMI_norm'] +
            self.weights['sentiment'] * df['Consumer_Sentiment_norm'] +
            self.weights['fed_rate'] * df['Fed_Rate_norm'] +
            self.weights['cpi'] * df['CPI_MoM_norm']
        )
        
        # Statistiques
        logger.info(f"FCI calculé")
        logger.info(f"  Moyenne: {df['FCI'].mean():.2f}")
        logger.info(f"  Écart-type: {df['FCI'].std():.2f}")
        logger.info(f"  Min: {df['FCI'].min():.2f} (date: {df['FCI'].idxmin().date()})")
        logger.info(f"  Max: {df['FCI'].max():.2f} (date: {df['FCI'].idxmax().date()})")
        
        return df
    
    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifie les régimes selon les seuils FCI.
        
        Args:
            df: DataFrame avec colonne FCI
        
        Returns:
            DataFrame avec colonne Regime
        """
        logger.info(f"Classification des régimes...")
        
        def get_regime(fci_value):
            if fci_value < self.thresholds['restrictive']:
                return 'RESTRICTIF'
            elif fci_value > self.thresholds['accommodating']:
                return 'ACCOMMODANT'
            else:
                return 'NEUTRE'
        
        df['Regime'] = df['FCI'].apply(get_regime)
        
        # Distribution
        regime_counts = df['Regime'].value_counts()
        total = len(df)
        
        logger.info(f"Régimes classifiés:")
        for regime, count in regime_counts.items():
            pct = count / total * 100
            logger.info(f"  {regime:12s}: {count:3d} mois ({pct:5.1f}%)")
        
        return df
    
    def get_current_state(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Obtient l'état actuel (dernière observation).
        
        Args:
            df: DataFrame avec FCI et régimes
        
        Returns:
            Dictionnaire avec état actuel
        """
        last_date = df.index[-1]
        last_row = df.iloc[-1]
        
        current_state = {
            'date': last_date,
            'ISM_PMI': last_row['ISM_PMI'],
            'Consumer_Sentiment': last_row['Consumer_Sentiment'],
            'Fed_Rate': last_row['Fed_Rate'],
            'CPI_MoM': last_row['CPI_MoM'],
            'FCI': last_row['FCI'],
            'Regime': last_row['Regime']
        }
        
        return current_state
    
    def plot_fci_evolution(self, df: pd.DataFrame, output_dir: str = 'results/figures') -> None:
        """
        Crée une visualisation complète du FCI.
        
        Args:
            df: DataFrame avec FCI et régimes
            output_dir: Répertoire de sortie
        """
        logger.info(f"Création des visualisations...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Évolution du FCI avec régimes
        ax1 = fig.add_subplot(gs[0:2, :])
        
        # Zones de régimes
        ax1.axhspan(self.thresholds['accommodating'], df['FCI'].max() + 5, 
                   alpha=0.2, color='green', label='Zone ACCOMMODANT')
        ax1.axhspan(self.thresholds['restrictive'], self.thresholds['accommodating'], 
                   alpha=0.2, color='gray', label='Zone NEUTRE')
        ax1.axhspan(df['FCI'].min() - 5, self.thresholds['restrictive'], 
                   alpha=0.2, color='red', label='Zone RESTRICTIF')
        
        # Ligne FCI
        ax1.plot(df.index, df['FCI'], linewidth=2, color='darkblue', label='FCI', zorder=5)
        
        # Seuils
        ax1.axhline(y=self.thresholds['accommodating'], color='green', 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=self.thresholds['restrictive'], color='red', 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # État actuel
        current = self.get_current_state(df)
        ax1.scatter(current['date'], current['FCI'], s=200, color='darkblue', 
                   zorder=10, edgecolors='white', linewidths=2)
        ax1.annotate(f"Actuel: {current['FCI']:.1f}\n{current['Regime']}", 
                    xy=(current['date'], current['FCI']),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_title('Financial Conditions Index (FCI) - Évolution Historique', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('FCI')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution FCI
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(df['FCI'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=self.thresholds['restrictive'], color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=self.thresholds['accommodating'], color='green', linestyle='--', linewidth=2)
        ax2.axvline(x=current['FCI'], color='darkblue', linestyle='-', linewidth=2, label='Actuel')
        ax2.set_title('Distribution du FCI', fontsize=12, fontweight='bold')
        ax2.set_xlabel('FCI')
        ax2.set_ylabel('Fréquence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Régimes en barres
        ax3 = fig.add_subplot(gs[2, 1])
        regime_counts = df['Regime'].value_counts()
        colors = {'RESTRICTIF': '#E63946', 'NEUTRE': '#F4A261', 'ACCOMMODANT': '#2A9D8F'}
        ax3.bar(regime_counts.index, regime_counts.values, 
               color=[colors[r] for r in regime_counts.index])
        ax3.set_title('Distribution des Régimes', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Nombre de Mois')
        for i, v in enumerate(regime_counts.values):
            ax3.text(i, v + 2, str(v), ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Indicateurs normalisés
        ax4 = fig.add_subplot(gs[3, :])
        ax4.plot(df.index, df['ISM_PMI_norm'], label='ISM PMI', alpha=0.7)
        ax4.plot(df.index, df['Consumer_Sentiment_norm'], label='Consumer Sentiment', alpha=0.7)
        ax4.plot(df.index, df['Fed_Rate_norm'], label='Fed Rate', alpha=0.7)
        ax4.plot(df.index, df['CPI_MoM_norm'], label='CPI MoM', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('Indicateurs Normalisés (Z-scores)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Z-score')
        ax4.legend(loc='upper left', ncol=4)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Financial Conditions Index (FCI) - Analyse Complète', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Sauvegarder
        output_path = Path(output_dir) / 'fci_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {output_path}")
        plt.close()
    
    def save_results(self, df: pd.DataFrame, output_dir: str = 'results/tables') -> None:
        """
        Sauvegarde les résultats du FCI.
        
        Args:
            df: DataFrame avec FCI
            output_dir: Répertoire de sortie
        """
        logger.info(f"Sauvegarde des résultats...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Série complète FCI
        fci_path = Path(output_dir) / 'fci_timeseries.csv'
        df[['FCI', 'Regime']].to_csv(fci_path)
        logger.info(f"{fci_path}")
        
        # 2. État actuel
        current = self.get_current_state(df)
        current_df = pd.DataFrame([current])
        current_path = Path(output_dir) / 'fci_current_state.csv'
        current_df.to_csv(current_path, index=False)
        logger.info(f"{current_path}")
        
        # 3. Statistiques par régime
        stats = df.groupby('Regime')['FCI'].agg(['count', 'mean', 'std', 'min', 'max'])
        stats_path = Path(output_dir) / 'fci_regime_statistics.csv'
        stats.to_csv(stats_path)
        logger.info(f"{stats_path}")
    
    def run_full_analysis(self) -> pd.DataFrame:
        """
        Exécute l'analyse complète du FCI.
        
        Returns:
            DataFrame avec FCI et régimes
        """
  
        logger.info("FINANCIAL CONDITIONS INDEX (FCI) - ANALYSE COMPLÈTE")
        
        # 1. Charger données
        df = self.load_data()
        
        # 2. Normaliser
        df = self.normalize_indicators(df)
        
        # 3. Calculer FCI
        df = self.calculate_fci(df)
        
        # 4. Classifier régimes
        df = self.classify_regime(df)
        
        # 5. État actuel
        current = self.get_current_state(df)
        logger.info(f"  Date:               {current['date'].date()}")
        logger.info(f"  ISM PMI:            {current['ISM_PMI']:.1f}")
        logger.info(f"  Consumer Sentiment: {current['Consumer_Sentiment']:.1f}")
        logger.info(f"  Fed Rate:     {current['Fed_Rate']:.2f}%")
        logger.info(f"  CPI MoM:            {current['CPI_MoM']:.2f}%")
        logger.info(f"FCI:             {current['FCI']:.2f}")
        logger.info(f"RÉGIME:          {current['Regime']}")
        
        # 6. Visualiser
        self.plot_fci_evolution(df)
        
        # 7. Sauvegarder
        self.save_results(df)
        

        logger.info("ANALYSE FCI TERMINÉE")
        logger.info("Fichiers générés:")
        logger.info("  • results/figures/fci_evolution.png")
        logger.info("  • results/tables/fci_timeseries.csv")
        logger.info("  • results/tables/fci_current_state.csv")
        logger.info("  • results/tables/fci_regime_statistics.csv")
        
        return df


if __name__ == "__main__":
    """Exécution de l'analyse FCI."""
    
    try:
        fci = FinancialConditionsIndex()
        df_fci = fci.run_full_analysis()
        
        logger.info("FCI prêt pour le BLOC 1 du rapport !")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()