"""Evaluation helpers for ShelfVision.

The package contains lightweight bbox metrics, error visualization tools,
automatic pipeline recommendation and multi-model comparison reports.
"""

from .compare_models import compare_models
from .metrics import evaluate_predictions_file
from .recommend_model import RecommendationWeights, recommend_best_model

__all__ = [
    "evaluate_predictions_file",
    "RecommendationWeights",
    "recommend_best_model",
    "compare_models",
]
