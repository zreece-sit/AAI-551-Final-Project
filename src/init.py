"""
Music Preference Predictor Package
This package provides tools for predicting music genre preferences
based on user demographics.
"""

__version__ = "1.0.0"

from .data_loader import DataLoader
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .predictor import MusicPredictor
from .visualizer import Visualizer

__all__ = [
    'DataLoader',
    'DataProcessor',
    'ModelTrainer',
    'MusicPredictor',
    'Visualizer'
]
