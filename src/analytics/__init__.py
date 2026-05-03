"""Analytical modules for ShelfVision.

The package contains additional retail-oriented analytics built on top of
model predictions: shelf density analysis, zone statistics and visual reports.
"""

from .density import analyze_density_file

__all__ = ["analyze_density_file"]
