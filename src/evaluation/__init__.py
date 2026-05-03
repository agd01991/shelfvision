"""Evaluation helpers for ShelfVision.

The package contains lightweight bbox metrics, error visualization tools
and automatic pipeline recommendation for predictions saved by run_inference.py.
"""

from .metrics import evaluate_predictions_file
from .recommend_model import RecommendationWeights, recommend_best_model

__all__ = [
    "evaluate_predictions_file",
    "RecommendationWeights",
    "recommend_best_model",
]
