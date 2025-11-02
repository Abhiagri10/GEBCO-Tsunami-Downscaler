"""GEBCO-Tsunami-Downscaler"""
__version__ = "1.2.0"
__author__ = "Abhishek"

from .downscaler import BathymetryProcessor, main
__all__ = ['BathymetryProcessor', 'main']
