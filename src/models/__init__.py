"""Models package for custom language model implementations.

This package provides custom language model implementations that can be
used with Hydra's _target_ instantiation pattern.
"""

from src.models.local_model import LocalLanguageModel

__all__ = ["LocalLanguageModel"]
