"""Simulation engines for different execution strategies.

This module provides engine implementations that control how simulation
steps are executed (sequential, parallel, etc.).
"""

from src.simulation.engines.base import BaseEngine

__all__ = ["BaseEngine"]
