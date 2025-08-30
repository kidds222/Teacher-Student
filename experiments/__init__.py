"""
Only contains core runtime files
"""

# Simplified runtime system
from .run import main, SimpleTrainer

__all__ = [
    'main',        # Main runtime function
    'SimpleTrainer'  # Simple trainer class
] 