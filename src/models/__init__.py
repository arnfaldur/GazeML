"""Model definitions (one class per file) to define NN architectures."""
from .elg import ELG
from .dpg import DPG
from .eye_tracker import EyeTracker

__all__ = ('ELG', 'DPG', 'EyeTracker')
