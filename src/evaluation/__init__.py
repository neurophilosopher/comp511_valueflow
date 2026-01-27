"""Evaluation package for probe-based agent measurement.

This package provides a probe system that queries agents at configurable
intervals to measure metrics defined in evaluation configs.

Key design: Probes call agent.act() but NOT agent.observe(), so probe
interactions don't affect agent memory or future behavior.
"""

from src.evaluation.probe_runner import ProbeRunner
from src.evaluation.probes import BooleanProbe, CategoricalProbe, NumericProbe, Probe, create_probe

__all__ = [
    "Probe",
    "CategoricalProbe",
    "NumericProbe",
    "BooleanProbe",
    "ProbeRunner",
    "create_probe",
]
