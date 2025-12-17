"""
tests/test_predictor.py - Tests for prediction module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.predictor import MusicPredictor


class TestMusicPredictor:
    """Test suite for MusicPredictor class."""
    
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor with mocked components."""
        with patch('src.predictor.tf.keras.models.load_model') as mock_load, \
             patch('src.predictor.joblib.load') as mock_joblib:
            
            # Mock model
            mock_model = Mock()
            mock_model.input_shape = (None, 20)
            mock_model.predict = Mock(return_value=np.random.rand(1, 10))
            mock_load.return_value = mock_model
            
            # Mock artifacts
            mock_scaler = Mock()
            mock_scaler.transform = Mock(return_value=np.zeros((1, 5)))
            
            mock_mlb = Mock()
            mock_mlb.classes_ = np.array(['rock', 'pop', 'jazz', 'metal', 'indie', 
                                         'electronic', 'hip hop', 'country', 'blues', 'classical'])
            
            mock_col_info = {
                'num_cols': ['age', 'registered'],
                'cat_cols': ['gender', 'country'],
                'X_cat_columns': ['gender_m', 'country_US', 'country_UK'],
                'genre_classes': list(mock_mlb.classes_)
            }
            
            mock_joblib.side_effect = [mock_scaler, mock_mlb, mock_col_info]
            
            # Create predictor
            predictor = MusicPredictor('fake_model.h5', 'fake_artifacts')
            predictor.model = mock_model
            predictor.scaler = mock_scaler
            predictor.mlb = mock_mlb
            
            return predictor
    
    def test_predictor_initialization_missing_model(self):
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError):
            MusicPredictor('nonexistent_model.h5', 'nonexistent_dir')
    
    def test_predict_returns_list(self, mock_predictor):
        """Test that predict returns a list of tuples."""
        user = {'age': 25, 'gender': 'm', 'country': 'US', 'registered': 2010}
        predictions = mock_predictor.predict(user, top_k=5)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 5
        
        # Check structure of predictions
        if len(predictions) > 0:
            assert isinstance(predictions[0], tuple)
            assert len(predictions[0]) == 2
            assert isinstance(predictions[0][0], str)  # genre name
            assert isinstance(predictions[0][1], float)  # probability
    
    def test_predict_with_threshold(self, mock_predictor):
        """Test prediction with probability threshold."""
        user = {'age': 25, 'gender': 'm', 'country': 'US', 'registered': 2010}
        
        # Mock high probabilities
        mock_predictor.model.predict = Mock(
            return_value=np.array([[0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.0]])
        )
        
        predictions = mock_predictor.predict(user, top_k=5, threshold=0.6)
        
        # Should only return predictions above threshold
        for _, prob in predictions:
            assert prob >= 0.6
    
    def test_predict_all_probs(self, mock_predictor):
        """Test getting all genre probabilities."""
        user = {'age': 25, 'gender': 'm', 'country': 'US', 'registered': 2010}
        all_probs = mock_predictor.predict_all_probs(user)
        
        assert isinstance(all_probs, dict)
        assert len(all_probs) == 10  # Number of genres
        
        # Check all values are probabilities
        for genre, prob in all_probs.items():
            assert isinstance(genre, str)
            assert 0 <= prob <= 1
    
    def test_str_representation(self, mock_predictor):
        """Test __str__ method."""
        str_repr = str(mock_predictor)
        assert "MusicPredictor" in str_repr
    
    def test_repr_representation(self, mock_predictor):
        """Test __repr__ method."""
        repr_str = repr(mock_predictor)
        assert "MusicPredictor" in repr_str
        assert "num_genres" in repr_str


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
