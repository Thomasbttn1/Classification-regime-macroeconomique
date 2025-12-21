"""
Module de collecte des données économiques et financières

Ce module contient les fonctions pour collecter:
- Les 4 indicateurs économiques (ISM PMI, Consumer Sentiment, Fed Rate, CPI)
- Les données de marché (S&P 500 pour les rendements futurs)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collect_economic_data import collect_economic_data
    #from .collect_market_data import collect_market_data
    #from .data_quality_checks import check_data_quality

__all__ = [
    'collect_economic_data',
    'collect_market_data',
    'check_data_quality'
]
