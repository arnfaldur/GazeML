"""Data-source definitions (one class per file)."""
from .eyelink import EyeLink
from .frames import FramesSource
from .hdf5 import HDF5Source
from .unityeyes import UnityEyes
from .video import Video
from .webcam import Webcam
__all__ = ('EyeLink', 'FramesSource', 'HDF5Source', 'UnityEyes', 'Video', 'Webcam')
