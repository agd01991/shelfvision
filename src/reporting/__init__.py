"""Reporting helpers for ShelfVision.

The package contains tools for assembling presentation-ready mini reports
from inference, evaluation, comparison and density analysis artifacts.
"""

from .mini_report import build_mini_report
from .segmentation_identification_report import generate_segmentation_identification_report

__all__ = ["build_mini_report", "generate_segmentation_identification_report"]
