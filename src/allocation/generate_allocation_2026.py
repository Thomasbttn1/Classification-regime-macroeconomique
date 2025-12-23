"""
Génération de l'Allocation d'Actifs pour 2026 - BLOC 3

Ce script génère la proposition d'allocation stratégique pour 2026
basée sur le régime macroéconomique prédit par le modèle.

L'allocation est proposée sur 8 classes d'actifs selon les règles
définies dans la stratégie de backtesting.

Usage:
    python src/allocation/generate_allocation_2026.py
"""

from pathlib import Path
from typing import Dict
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class AllocationGenerator2026:
    """Génère l'allocation d'actifs pour 2026."""
    
    def __init__(self):
        """
        Initialise le générateur d'allocation.
        
        Utilise les règles d'allocation optimisées du backtesting.
        """
        # Règles d'allocation par régime (issues du backtesting optimisé)
        self.allocation_rules = {
            'RESTRICTIF': {
                'SPX': 0.20,      # Actions US
                'SX5E': 0.15,     # Actions Euro
                'USGG10YR': 0.20, # Taux longs US 10Y
                'USGG2YR': 0.12,  # Taux courts US 2Y
                'GDBR10': 0.15,   # Taux longs Allemands
                'GDBR2': 0.08,    # Taux courts Allemands
                'GFRN10': 0.06,   # Taux longs Français
                'ESTRON': 0.04    # Marché monétaire Euro
            },
            'NEUTRE': {
                'SPX': 0.50,
                'SX5E': 0.20,
                'USGG10YR': 0.10,
                'USGG2YR': 0.06,
                'GDBR10': 0.06,
                'GDBR2': 0.04,
                'GFRN10': 0.02,
                'ESTRON': 0.02
            },
            'ACCOMMODANT': {
                'SPX': 0.52,      # 52% actions US
                'SX5E': 0.23,     # 23% actions Euro
                'USGG10YR': 0.10, # 10% Taux longs US 10Y
                'USGG2YR': 0.05,  # 5% Taux courts US 2Y
                'GDBR10': 0.05,   # 5% Taux longs Allemands
                'GDBR2': 0.03,    # 3% Taux courts Allemands
                'GFRN10': 0.01,   # 1% Taux longs Français
                'ESTRON': 0.01    # 1% Marché monétaire Euro
            }
        }
        
        # Noms complets des actifs
        self.asset_names = {
            'SPX': 'Actions US (SPX Index)',
            'SX5E': 'Actions EURO (SX5E Index)',
            'USGG10YR': 'Taux longs US 10 ans (USGG10YR Index)',
            'USGG2YR': 'Taux courts US 2 ans (USGG2YR Index)',
            'GDBR10': 'Taux longs Allemands (GDBR10 Index)',
            'GDBR2': 'Taux courts Allemands (GDBR2 Index)',
            'GFRN10': 'Taux longs Français (GFRN10 Index)',
            'ESTRON': 'Marché monétaire EURO (ESTRON Index)'
        }
        
        logger.info("✓ Générateur d'allocation initialisé")
    
    def load_regime_prediction(
        self,
        prediction_path: str = 'results/tables/regime_prediction_2026.csv'
    ) -> str:
        """
        Charge la prédiction de régime pour 2026.
        
        Args:
            prediction_path: Chemin vers la prédiction
        
        Returns:
            Régime prédit
        """
        logger.info(f"\n📂 Chargement de la prédiction de régime...")
        
        prediction_df = pd.read_csv(prediction_path)
        regime_predit = prediction_df['regime_predit_2026'].iloc[0]
        
        logger.info(f"✓ Régime prédit pour 2026: {regime_predit}")
        
        return regime_predit
    
    def generate_allocation(self, regime: str) -> pd.DataFrame:
        """
        Génère l'allocation basée sur le régime.
        
        Args:
            regime: Régime macroéconomique
        
        Returns:
            DataFrame avec allocation
        """
        logger.info(f"\n🎯 Génération de l'allocation pour régime {regime}...")
        
        # Récupérer l'allocation
        allocation = self.allocation_rules[regime]
        
        # Créer DataFrame
        allocation_data = []
        for asset_code, weight in allocation.items():
            allocation_data.append({
                'Classe d\'Actif': self.asset_names[asset_code],
                'Code': asset_code,
                'Allocation 2026': f"{weight:.1%}",
                'Allocation_Numeric': weight
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        logger.info(f"✓ Allocation générée")
        
        return allocation_df
    
    def generate_justification(self, regime: str, allocation_df: pd.DataFrame) -> Dict[str, str]:
        """
        Génère les justifications pour chaque ligne d'allocation.
        
        Args:
            regime: Régime macroéconomique
            allocation_df: DataFrame avec allocation
        
        Returns:
            Dictionnaire avec justifications
        """
        logger.info(f"\n📝 Génération des justifications...")
        
        # Justifications par régime et type d'actif
        justifications_templates = {
            'RESTRICTIF': {
                'actions': "Réduction de l'exposition actions ({total_equity:.0%}) face au risque de ralentissement économique. Position défensive privilégiée.",
                'obligations_longues': "Surpondération des obligations longues ({total_long_bonds:.0%}) pour bénéficier de la baisse probable des taux en contexte restrictif.",
                'obligations_courtes': "Maintien d'obligations courtes ({total_short_bonds:.0%}) pour la liquidité et flexibilité du portefeuille.",
                'cash': "Position cash ({cash:.0%}) pour préserver le capital et saisir opportunités."
            },
            'NEUTRE': {
                'actions': "Allocation équilibrée en actions ({total_equity:.0%}) cohérente avec une croissance modérée anticipée.",
                'obligations_longues': "Position modérée en obligations longues ({total_long_bonds:.0%}) pour diversification.",
                'obligations_courtes': "Obligations courtes ({total_short_bonds:.0%}) pour gérer la duration et profiter du portage.",
                'cash': "Cash minimal ({cash:.0%}) car environnement favorable au déploiement du capital."
            },
            'ACCOMMODANT': {
                'actions': "Forte exposition actions ({total_equity:.0%}) pour capter la croissance économique favorable. Position opportuniste.",
                'obligations_longues': "Sous-pondération obligations longues ({total_long_bonds:.0%}) car rendement/risque moins attractif en phase expansive.",
                'obligations_courtes': "Minimum en obligations courtes ({total_short_bonds:.0%}) pour concentration sur actifs risqués.",
                'cash': "Cash minimal ({cash:.0%}) pour maximiser l'exposition au marché actions."
            }
        }
        
        # Calculer les agrégats
        total_equity = allocation_df[allocation_df['Code'].isin(['SPX', 'SX5E'])]['Allocation_Numeric'].sum()
        total_long_bonds = allocation_df[allocation_df['Code'].isin(['USGG10YR', 'GDBR10', 'GFRN10'])]['Allocation_Numeric'].sum()
        total_short_bonds = allocation_df[allocation_df['Code'].isin(['USGG2YR', 'GDBR2'])]['Allocation_Numeric'].sum()
        cash = allocation_df[allocation_df['Code'] == 'ESTRON']['Allocation_Numeric'].iloc[0]
        
        # Générer justifications
        justifications = {}
        templates = justifications_templates[regime]
        
        for key, template in templates.items():
            justifications[key] = template.format(
                total_equity=total_equity,
                total_long_bonds=total_long_bonds,
                total_short_bonds=total_short_bonds,
                cash=cash
            )
        
        logger.info(f"✓ Justifications générées")
        
        return justifications
    
    def create_allocation_table(
        self,
        allocation_df: pd.DataFrame,
        justifications: Dict[str, str],
        regime: str
    ) -> pd.DataFrame:
        """
        Crée le tableau final avec justifications.
        
        Args:
            allocation_df: DataFrame avec allocations
            justifications: Justifications
            regime: Régime
        
        Returns:
            DataFrame formaté pour le rapport
        """
        # Mapper les justifications
        justification_map = {
            'SPX': justifications['actions'],
            'SX5E': justifications['actions'],
            'USGG10YR': justifications['obligations_longues'],
            'GDBR10': justifications['obligations_longues'],
            'GFRN10': justifications['obligations_longues'],
            'USGG2YR': justifications['obligations_courtes'],
            'GDBR2': justifications['obligations_courtes'],
            'ESTRON': justifications['cash']
        }
        
        allocation_df['Justification'] = allocation_df['Code'].map(justification_map)
        
        # Sélectionner colonnes pour rapport
        final_df = allocation_df[['Classe d\'Actif', 'Allocation 2026', 'Justification']]
        
        return final_df
    
    def plot_allocation(
        self,
        allocation_df: pd.DataFrame,
        regime: str,
        output_path: str = 'results/figures/allocation_2026.png'
    ) -> None:
        """
        Crée une visualisation de l'allocation.
        
        Args:
            allocation_df: DataFrame avec allocation
            regime: Régime
            output_path: Chemin de sortie
        """
        logger.info(f"\n📊 Création de la visualisation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Camembert par actif
        colors = plt.cm.Set3(range(len(allocation_df)))
        ax1.pie(
            allocation_df['Allocation_Numeric'],
            labels=allocation_df['Code'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title(f'Allocation 2026 par Actif\n(Régime {regime})', 
                     fontsize=12, fontweight='bold')
        
        # 2. Barres horizontales
        allocation_sorted = allocation_df.sort_values('Allocation_Numeric', ascending=True)
        y_pos = np.arange(len(allocation_sorted))
        
        ax2.barh(y_pos, allocation_sorted['Allocation_Numeric'] * 100, color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(allocation_sorted['Code'])
        ax2.set_xlabel('Allocation (%)')
        ax2.set_title(f'Allocation 2026 Détaillée\n(Régime {regime})', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Ajouter valeurs
        for i, v in enumerate(allocation_sorted['Allocation_Numeric']):
            ax2.text(v * 100 + 1, i, f'{v:.1%}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Sauvegarder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Graphique sauvegardé: {output_path}")
        plt.close()
    
    def save_allocation(
        self,
        final_df: pd.DataFrame,
        regime: str,
        output_path: str = 'results/tables/allocation_2026.csv'
    ) -> None:
        """
        Sauvegarde l'allocation finale.
        
        Args:
            final_df: DataFrame final
            regime: Régime
            output_path: Chemin de sortie
        """
        logger.info(f"\n💾 Sauvegarde de l'allocation...")
        
        # Ajouter métadonnées
        metadata = pd.DataFrame([{
            'Date de génération': pd.Timestamp.now(),
            'Régime prédit 2026': regime,
            'Méthode': 'Allocation stratégique basée sur classification ML des régimes macroéconomiques'
        }])
        
        # Sauvegarder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# ALLOCATION STRATEGIQUE 2026\n")
            metadata.to_csv(f, index=False)
            f.write("\n")
            final_df.to_csv(f, index=False)
        
        logger.info(f"✓ Allocation sauvegardée: {output_path}")
    
    def display_allocation(self, final_df: pd.DataFrame, regime: str) -> None:
        """
        Affiche l'allocation de manière formatée.
        
        Args:
            final_df: DataFrame final
            regime: Régime
        """
        logger.info("\n" + "="*70)
        logger.info("📋 ALLOCATION STRATÉGIQUE 2026")
        logger.info("="*70)
        logger.info(f"\n🏷️  Régime prédit: {regime}")
        logger.info(f"\n📊 ALLOCATION PROPOSÉE:\n")
        
        print(final_df.to_string(index=False))
        
        # Résumé
        logger.info("\n" + "="*70)
        logger.info("📈 RÉSUMÉ")
        logger.info("="*70)
        
        # Calculer agrégats
        allocations = final_df['Allocation 2026'].str.rstrip('%').astype(float) / 100
        codes = final_df['Classe d\'Actif'].str.extract(r'\((\w+)')[0]
        
        total_actions = allocations[codes.isin(['SPX', 'SX5E'])].sum()
        total_taux = allocations[~codes.isin(['SPX', 'SX5E'])].sum()
        
        logger.info(f"  Total Actions:     {total_actions:.1%}")
        logger.info(f"  Total Obligations: {total_taux:.1%}")
    
    def run(self) -> pd.DataFrame:
        """
        Exécute la génération complète de l'allocation.
        
        Returns:
            DataFrame avec allocation finale
        """
        logger.info("\n" + "="*70)
        logger.info("GÉNÉRATION DE L'ALLOCATION STRATÉGIQUE 2026")
        logger.info("="*70)
        
        # 1. Charger prédiction
        regime = self.load_regime_prediction()
        
        # 2. Générer allocation
        allocation_df = self.generate_allocation(regime)
        
        # 3. Générer justifications
        justifications = self.generate_justification(regime, allocation_df)
        
        # 4. Créer tableau final
        final_df = self.create_allocation_table(allocation_df, justifications, regime)
        
        # 5. Visualiser
        self.plot_allocation(allocation_df, regime)
        
        # 6. Afficher
        self.display_allocation(final_df, regime)
        
        # 7. Sauvegarder
        self.save_allocation(final_df, regime)
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALLOCATION 2026 GÉNÉRÉE")
        logger.info("="*70)
        logger.info("\n📁 Fichiers générés:")
        logger.info("  • results/tables/allocation_2026.csv")
        logger.info("  • results/figures/allocation_2026.png")
        
        return final_df


if __name__ == "__main__":
    """Exécution de la génération d'allocation."""
    
    try:
        generator = AllocationGenerator2026()
        allocation_df = generator.run()
        
        logger.info("\n🎉 Allocation 2026 prête pour le rapport BLOC 3 !")
        logger.info("\n📝 Copiez le tableau dans votre rapport Word")
        logger.info("   et incluez le graphique en annexe.")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()