"""Model training module for music genre preference prediction. This module contains the ModelTrainer class for building and training neural network models."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict


class ModelTrainer:
    """
    Handles neural network model creation and training.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output classes (genres)
        model (tf.keras.Model): The neural network model
        history: Training history object
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize ModelTrainer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.history = None
    
    def build_model(self, hidden_layers: list = [256, 128, 64],
                   dropout_rates: list = [0.3, 0.2, 0.1]) -> None:
        """
        Build a multi-layer neural network for multi-label classification.
        
        Args:
            hidden_layers: List of hidden layer sizes
            dropout_rates: List of dropout rates for each layer
        """
        # Input layer
        inp = layers.Input(shape=(self.input_dim,), name='input_layer')
        x = inp
        
        # Hidden layers with dropout using enumerate
        for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
            x = layers.Dense(units, activation='relu', 
                           name=f'hidden_{i+1}')(x)
            x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer (sigmoid for multi-label)
        out = layers.Dense(self.output_dim, activation='sigmoid',
                          name='output_layer')(x)
        
        # Create and compile model
        self.model = models.Model(inputs=inp, outputs=out, name='genre_predictor')
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        print(f"Input dimension: {self.input_dim}")
        print(f"Output dimension: {self.output_dim}")
    
    def train(self, X: np.ndarray, Y: np.ndarray,
             test_size: float = 0.3,
             val_size: float = 0.5,
             epochs: int = 50,
             batch_size: int = 64,
             model_save_path: str = 'models/multilabel_model.h5',
             random_state: int = 42) -> Dict:
        """
        Train the model with train/validation/test split.
        
        Args:
            X: Feature matrix
            Y: Label matrix
            test_size: Proportion for test set
            val_size: Proportion of remaining for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save best model
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test data splits
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Split data: train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        
        # Split temp: val + test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        
        # Ensure float32 dtype
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Create callbacks
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, model_checkpoint],
            verbose=2
        )
        
        print("\nTraining complete!")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
