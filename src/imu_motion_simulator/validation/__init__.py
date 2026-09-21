"""Reality-baseline validation kept outside production acceptance."""

from .imucoco import audit_imucoco, benchmark_imucoco

__all__ = ['audit_imucoco', 'benchmark_imucoco']
