"""Probability modeling module."""

from footbe_trader.modeling.interfaces import IModel, PredictionResult
from footbe_trader.modeling.placeholder import PlaceholderModel

__all__ = [
    "IModel",
    "PredictionResult",
    "PlaceholderModel",
]
