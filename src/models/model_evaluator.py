"""
Module d'évaluation pour la comparaison des modèles.
Ce module évalue et compare les performances des deux modèles, Random Forest et Gradient Boosting, sur la
classification des régimes macro
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from random_forest_model import RandomForestModel
from gradient_boosting_model import GradientBoostingModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger : logging.Logger = logging.getLogger(__name__)

# Style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ModelEvaluator:
    def __init__(self, output_dir : str = 'results') -> None:
        self.output_dir : Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir : Path = self.output_dir / 'figures'
        self.tables_dir : Path = self.output_dir / 'tables'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluateur initialisé")

    def load_test_data(self, splits_dir : str = 'data/processed/splits') -> Dict[str, Any]:
        """
        Charges les splits sauvegardés
        """
        logger.info("Chargement des données de splits")

        splits_path : Path = Path(splits_dir)

        data: Dict[str, Any] = {
            'X_test': pd.read_csv(splits_path / 'X_test.csv', index_col=0, parse_dates=True),
            'y_test': pd.read_csv(splits_path / 'y_test.csv', index_col=0, parse_dates=True).squeeze()
        }
        logger.info("Données chargées")
        return data
    def load_and_evaluate(self, model_class, model_name : str, model_path : str, X_test : pd.DataFrame, y_test : pd.Series) -> Dict[str, Any]:
        """
        Charge un modèle déjà sauvegardé et l'évalue
        """
        logger.info("-"*70)
        logger.info(f"{model_name}")
        logger.info("-"*70)

        #vérifier que le fichier existe
        if not Path(model_path).exists():
            raise FileNotFoundError("Modèle non trouvé. Executez les fichier .py")
        
        #Charger le modèle
        logger.info("Chargement du modèle")
        model = model_class()
        model.load(model_path)
        logger.info("Modèle chargé")

        #Nettoyer les NaN
        X_test_clean = X_test.fillna(X_test.median())

        #Prédictions
        logger.info(f"Prédictions sur {len(X_test_clean)} samples...")
        y_pred: np.ndarray = model.predict(X_test_clean)
        y_proba: np.ndarray = model.predict_proba(X_test_clean)

        # Métriques
        test_acc: float = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        cm: np.ndarray = confusion_matrix(y_test, y_pred)

        # Feature importance
        try:
            importance: pd.DataFrame = model.get_feature_importance(top_n=10)
        except TypeError:
            importance: pd.DataFrame = model.get_feature_importance()
            importance = importance.head(10)
        
        logger.info(f"✓ Accuracy: {test_acc:.2%} | F1-Score: {f1:.2%}")
        
        return {
            'model': model,
            'model_name': model_name,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'importance': importance
        }
    
    def plot_confusion_matrices(self, results : Dict[str, Dict[str, Any]], y_test : pd.Series) -> None:
        """
        Créer les matrices de confusion côte à côte
        """
        logger.info("Création des confusion matrices...")
        
        n_models: int = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        classes: List[str] = sorted(y_test.unique())
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm: np.ndarray = result['confusion_matrix']
            cm_norm: np.ndarray = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_norm,
                annot=cm,
                fmt='d',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                ax=axes[idx],
                cbar_kws={'label': 'Proportion'},
                vmin=0,
                vmax=1
            )
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.2%}')
            axes[idx].set_ylabel('Vrai Label')
            axes[idx].set_xlabel('Prédiction')
        
        plt.tight_layout()
        output_path: Path = self.figures_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sauvegardé: {output_path}")
        plt.close()

    def plot_feature_importance(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Compare l'importance des features."""
        logger.info("Comparaison des feature importances...")
        
        n_models: int = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            importance: pd.DataFrame = result['importance']
            
            # Convertir en listes
            features = importance['feature'].tolist()
            importances = importance['importance'].tolist()
            
            # Utiliser des indices numériques au lieu des strings
            y_pos = np.arange(len(features))
            
            axes[idx].barh(
                y_pos,              # ← Indices numériques !
                importances,
                color='steelblue'
            )
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features)  # ← Labels après
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name}\nTop 10 Features')
            axes[idx].invert_yaxis()
            
            for i, v in enumerate(importances):
                axes[idx].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path: Path = self.figures_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sauvegardé: {output_path}")
        plt.close()

    def create_comparison_table(self, results : Dict[str, Dict[str, Any]]) -> pd.DataFrame :
        logger.info("Création du tableau de comparaison...")
        
        comparison_data: List[Dict[str, Any]] = []
        
        for model_name, result in results.items():
            row: Dict[str, Any] = {
                'Modèle': model_name,
                'Test Accuracy': f"{result['accuracy']:.2%}",
                'Precision': f"{result['precision']:.2%}",
                'Recall': f"{result['recall']:.2%}",
                'F1-Score': f"{result['f1']:.2%}"
            }
            comparison_data.append(row)
        
        comparison_df: pd.DataFrame = pd.DataFrame(comparison_data)
        
        output_path: Path = self.tables_dir / 'model_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Sauvegardé: {output_path}")
        
        return comparison_df

    def create_per_class_metrics(self, results : Dict[str, Dict[str, Any]], y_test : pd.Series) -> pd.DataFrame:
        """
        Créer les métriques par classe
        """ 
        logger.info("Création des métriques par classe...")
        
        classes: List[str] = sorted(y_test.unique())
        rows: List[Dict[str, Any]] = []
        
        for model_name, result in results.items():
            y_pred: np.ndarray = result['y_pred']
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, labels=classes, zero_division=0
            )
            
            for i, classe in enumerate(classes):
                row: Dict[str, Any] = {
                    'Modèle': model_name,
                    'Classe': classe,
                    'Precision': f"{precision[i]:.2%}",
                    'Recall': f"{recall[i]:.2%}",
                    'F1-Score': f"{f1[i]:.2%}",
                    'Support': int(support[i])
                }
                rows.append(row)
        
        metrics_df: pd.DataFrame = pd.DataFrame(rows)
        
        output_path: Path = self.tables_dir / 'per_class_metrics.csv'
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Sauvegardé: {output_path}")
        
        return metrics_df
    
if __name__ == "__main__":
    """Évaluation complète - Charge et compare les modèles sauvegardés."""
    
    try:
        logger.info("\n" + "="*70)
        logger.info("ÉVALUATION DES MODÈLES SAUVEGARDÉS")
        logger.info("="*70)
        
        # Initialiser
        evaluator: ModelEvaluator = ModelEvaluator()
        
        # Charger les données de test
        data: Dict[str, Any] = evaluator.load_test_data()
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Chemins des modèles
        rf_path: str = 'models/saved_models/randomforestmodel.pkl'
        gb_path: str = 'models/saved_models/gradientboostingmodel.pkl'
        
        # Charger et évaluer Random Forest
        logger.info("\n1️⃣ Random Forest")
        rf_results: Dict[str, Any] = evaluator.load_and_evaluate(
            RandomForestModel, "Random Forest", rf_path, X_test, y_test
        )
        
        # Charger et évaluer Gradient Boosting
        logger.info("\n2️⃣ Gradient Boosting")
        gb_results: Dict[str, Any] = evaluator.load_and_evaluate(
            GradientBoostingModel, "Gradient Boosting", gb_path, X_test, y_test
        )
        
        # Combiner les résultats
        all_results: Dict[str, Dict[str, Any]] = {
            'Random Forest': rf_results,
            'Gradient Boosting': gb_results
        }
        
        # Créer les visualisations
        logger.info("\n" + "="*70)
        logger.info("3️⃣ GÉNÉRATION DES VISUALISATIONS")
        logger.info("="*70)
        
        evaluator.plot_confusion_matrices(all_results, y_test)
        evaluator.plot_feature_importance(all_results)
        
        # Créer les tableaux
        logger.info("\n" + "="*70)
        logger.info("4️⃣ GÉNÉRATION DES TABLEAUX")
        logger.info("="*70)
        
        comparison_df: pd.DataFrame = evaluator.create_comparison_table(all_results)
        metrics_df: pd.DataFrame = evaluator.create_per_class_metrics(all_results, y_test)
        
        # Afficher les résultats
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSULTATS")
        logger.info("="*70)
        
        logger.info("\n🏆 Comparaison Globale:")
        logger.info("\n" + comparison_df.to_string(index=False))
        
        logger.info("\n📈 Métriques par Classe:")
        logger.info("\n" + metrics_df.to_string(index=False))
        
        # Résumé final
        logger.info("\n" + "="*70)
        logger.info("✅ ÉVALUATION TERMINÉE")
        logger.info("="*70)
        logger.info("\n📁 Fichiers générés:")
        logger.info("  • results/figures/confusion_matrices.png")
        logger.info("  • results/figures/feature_importance.png")
        logger.info("  • results/tables/model_comparison.csv")
        logger.info("  • results/tables/per_class_metrics.csv")
        logger.info("\n🎉 Tous les résultats sont prêts pour votre rapport!")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ {e}")
        logger.info("\n💡 Solution: Entraînez d'abord les modèles:")
        logger.info("  python src/models/random_forest_model.py")
        logger.info("  python src/models/gradient_boosting_model.py")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
            


