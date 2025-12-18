""" predictor.py - Prediction module for new users """

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from typing import Dict, List, Tuple


class MusicPredictor:
    """ Makes genre predictions for new users based on demographics.    
    Uses composition with loaded model and preprocessing artifacts. """
    
    def __init__(self, model_path: str, artifacts_dir: str = 'artifacts'):
        """ Initialize predictor with saved model and artifacts """
       
        if not os.path.exists(model_path):  # loads model
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        
        # loads preprocessing artifacts
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
        """ Preprocesses user data to match training format """
        
        user_df = pd.DataFrame([user_dict])  # creates DataFrame from user dict
        
        # processes numeric features
        if self.num_cols:
            X_num = user_df[self.num_cols].fillna(0)
            X_num_scaled = pd.DataFrame(self.scaler.transform(X_num), columns=X_num.columns)
        else:
            X_num_scaled = pd.DataFrame()
        
        # processes categorical features
        if self.cat_cols:
            X_cat = pd.get_dummies(user_df[self.cat_cols].astype(str))
            
            # aligns with training columns
            cat_aligned = pd.DataFrame(0, index=[0], columns=self.X_cat_columns)
            for col in X_cat.columns:
                if col in cat_aligned.columns:
                    cat_aligned.loc[0, col] = X_cat.iloc[0][col]
        else:
            cat_aligned = pd.DataFrame()
        
        # combines features
        X_final = pd.concat([X_num_scaled.reset_index(drop=True), cat_aligned.reset_index(drop=True)], axis=1).values.astype(np.float32)
        
        return X_final
    
    def predict(self, user_dict: Dict, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """ Predicts top-K genres for a new user """
        
        X = self._preprocess_user(user_dict)  # preprocesses user data
        probs = self.model.predict(X, verbose=0)[0]  # gets predictions
        top_idx = np.argsort(probs)[-top_k:][::-1]  # gets top-k indices
        
        # filters by threshold using list comprehension
        predictions = [
            (self.genre_classes[i], float(probs[i]))
            for i in top_idx if probs[i] >= threshold
        ]
        
        # If no predictions above threshold --> return top-k anyway
        if not predictions:
            predictions = [
                (self.genre_classes[i], float(probs[i]))
                for i in top_idx
            ]
        
        return predictions  # returns list of (genre, probability) tuples
    
    def predict_all_probs(self, user_dict: Dict) -> Dict[str, float]:
        """ Gets probabilities for all genres """
        X = self._preprocess_user(user_dict)
        probs = self.model.predict(X, verbose=0)[0]
        
        return dict(zip(self.genre_classes, map(float, probs)))  # uses zip to combine genres with probabilities
    
    @classmethod
    def load_model(cls, model_path: str, artifacts_dir: str = 'artifacts'):
        """ Class method to load a predictor """
        
        return cls(model_path, artifacts_dir)  # Initialized MusicPredictor instance
    
    def __str__(self) -> str:
        """String representation."""
        return f"MusicPredictor(genres={len(self.genre_classes)})"
    
    def __repr__(self) -> str:
        return (f"MusicPredictor(num_genres={len(self.genre_classes)}, "
                f"num_features={self.model.input_shape[1]})")
