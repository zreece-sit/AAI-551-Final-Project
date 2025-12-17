"""
predictor.py - Prediction module for new users
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from typing import Dict, List, Tuple


class MusicPredictor:
    """
    Makes genre predictions for new users based on demographics.
    
    Uses composition with loaded model and preprocessing artifacts.
    
    Attributes:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        mlb: Fitted MultiLabelBinarizer
        col_info: Dictionary with column information
    """
    
    def __init__(self, model_path: str, artifacts_dir: str = 'artifacts'):
        """
        Initialize predictor with saved model and artifacts.
        
        Args:
            model_path: Path to saved Keras model
            artifacts_dir: Directory containing preprocessing artifacts
            
        Raises:
            FileNotFoundError: If required files don't exist
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        
        # Load preprocessing artifacts
        try:
            self.scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
            self.mlb = joblib.load(os.path.join(artifacts_dir, 'mlb.joblib'))
            self.col_info = joblib.load(os.path.join(artifacts_dir, 'cols_info.joblib'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Artifact file not found: {e}")
        
        self.num_cols = self.col_info['num_cols']
        self.cat_cols = self.col_info['cat_cols']
        self.X_cat_columns = self.col_info['X_cat_columns']
        self.genre_classes = self.col_info['genre_classes']
        
        print("MusicPredictor loaded successfully!")
        print(f"Model expects {self.model.input_shape[1]} features")
        print(f"Predicts {len(self.genre_classes)} genres")
    
    def _preprocess_user(self, user_dict: Dict) -> np.ndarray:
        """
        Preprocess user data to match training format.
        
        Args:
            user_dict: Dictionary with user demographics
            
        Returns:
            Preprocessed feature array
        """
        # Create DataFrame from user dict
        user_df = pd.DataFrame([user_dict])
        
        # Process numeric features
        if self.num_cols:
            X_num = user_df[self.num_cols].fillna(0)
            X_num_scaled = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=X_num.columns
            )
        else:
            X_num_scaled = pd.DataFrame()
        
        # Process categorical features
        if self.cat_cols:
            X_cat = pd.get_dummies(user_df[self.cat_cols].astype(str))
            
            # Align with training columns
            cat_aligned = pd.DataFrame(0, index=[0], columns=self.X_cat_columns)
            for col in X_cat.columns:
                if col in cat_aligned.columns:
                    cat_aligned.loc[0, col] = X_cat.iloc[0][col]
        else:
            cat_aligned = pd.DataFrame()
        
        # Combine features
        X_final = pd.concat(
            [X_num_scaled.reset_index(drop=True), 
             cat_aligned.reset_index(drop=True)],
            axis=1
        ).values.astype(np.float32)
        
        return X_final
    
    def predict(self, user_dict: Dict, top_k: int = 5, 
               threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Predict top-K genres for a new user.
        
        Args:
            user_dict: User demographics {'age': 25, 'gender': 'm', ...}
            top_k: Number of top genres to return
            threshold: Minimum probability threshold
            
        Returns:
            List of (genre, probability) tuples
        """
        # Preprocess user data
        X = self._preprocess_user(user_dict)
        
        # Get predictions
        probs = self.model.predict(X, verbose=0)[0]
        
        # Get top-k indices
        top_idx = np.argsort(probs)[-top_k:][::-1]
        
        # Filter by threshold using list comprehension
        predictions = [
            (self.genre_classes[i], float(probs[i]))
            for i in top_idx if probs[i] >= threshold
        ]
        
        # If no predictions above threshold, return top-k anyway
        if not predictions:
            predictions = [
                (self.genre_classes[i], float(probs[i]))
                for i in top_idx
            ]
        
        return predictions
    
    def predict_all_probs(self, user_dict: Dict) -> Dict[str, float]:
        """
        Get probabilities for all genres.
        
        Args:
            user_dict: User demographics
            
        Returns:
            Dictionary mapping genre to probability
        """
        X = self._preprocess_user(user_dict)
        probs = self.model.predict(X, verbose=0)[0]
        
        # Use zip to combine genres with probabilities
        return dict(zip(self.genre_classes, map(float, probs)))
    
    @classmethod
    def load_model(cls, model_path: str, artifacts_dir: str = 'artifacts'):
        """
        Class method to load a predictor.
        
        Args:
            model_path: Path to model file
            artifacts_dir: Path to artifacts directory
            
        Returns:
            Initialized MusicPredictor instance
        """
        return cls(model_path, artifacts_dir)
    
    def __str__(self) -> str:
        """String representation."""
        return f"MusicPredictor(genres={len(self.genre_classes)})"
    
    def __repr__(self) -> str:
        """Official string representation."""
        return (f"MusicPredictor(num_genres={len(self.genre_classes)}, "
                f"num_features={self.model.input_shape[1]})")


# =============================================================================
# visualizer.py - Visualization module
# =============================================================================

import matplotlib.pyplot as plt


class Visualizer:
    """
    Creates visualizations for music preference predictions.
    
    Attributes:
        predictor: MusicPredictor instance for making predictions
    """
    
    def __init__(self, predictor: MusicPredictor):
        """
        Initialize Visualizer with a predictor.
        
        Args:
            predictor: MusicPredictor instance
        """
        self.predictor = predictor
    
    def plot_top_k_predictions(self, user_dict: Dict, k: int = 5,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot horizontal bar chart of top-K predictions.
        
        Args:
            user_dict: User demographics dictionary
            k: Number of top genres to show
            figsize: Figure size (width, height)
        """
        # Get predictions
        predictions = self.predictor.predict(user_dict, top_k=k, threshold=0.0)
        
        # Extract genres and probabilities
        genres = [genre for genre, _ in predictions]
        probs = [prob for _, prob in predictions]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(genres[::-1], probs[::-1], color='skyblue', edgecolor='navy')
        plt.xlim(0, 1)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title(f'Top-{k} Genre Predictions for User', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add probability labels
        for i, (genre, prob) in enumerate(zip(genres[::-1], probs[::-1])):
            plt.text(prob + 0.02, i, f'{prob:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_genres(self, user_dict: Dict,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot probabilities for all genres.
        
        Args:
            user_dict: User demographics dictionary
            figsize: Figure size
        """
        # Get all probabilities
        all_probs = self.predictor.predict_all_probs(user_dict)
        
        # Sort by probability
        sorted_items = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        genres = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
        
        # Create plot
        plt.figure(figsize=figsize)
        colors = ['green' if p > 0.5 else 'orange' if p > 0.3 else 'red' 
                 for p in probs]
        
        plt.barh(range(len(genres)), probs, color=colors, alpha=0.7)
        plt.yticks(range(len(genres)), genres, fontsize=8)
        plt.xlabel('Probability', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title('All Genre Probabilities', fontsize=14, fontweight='bold')
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='High Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: tf.keras.callbacks.History,
                             figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Keras History object
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss Over Time')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy Over Time')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation."""
        return f"Visualizer(predictor={self.predictor})"


if __name__ == "__main__":
    print("Predictor and Visualizer modules loaded")
predictor_visualizer.py
Displaying predictor_visualizer.py.
