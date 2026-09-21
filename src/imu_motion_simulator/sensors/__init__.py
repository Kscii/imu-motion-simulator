"""Configurable virtual IMU derivation from canonical motion."""

from .convergence import convergence_report
from .derive import derive_ideal, derive_calibrated
from .layout import load_layout, load_profile

__all__ = ['derive_ideal', 'derive_calibrated', 'convergence_report',
           'load_layout', 'load_profile']
