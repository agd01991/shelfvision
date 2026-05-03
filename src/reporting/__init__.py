"""Reporting helpers for ShelfVision.

The package contains tools for assembling presentation-ready mini reports
from inference, evaluation, comparison and density analysis artifacts.
"""

from .mini_report import build_mini_report

__all__ = ["build_mini_report"]
