"""Evaluation helpers for ShelfVision.

The package contains lightweight bbox metrics and error visualization tools
for predictions saved by run_inference.py.
"""

from .metrics import evaluate_predictions_file

__all__ = ["evaluate_predictions_file"]
