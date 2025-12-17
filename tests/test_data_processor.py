"""
tests/test_data_processor.py - Tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from collections import Counter


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor(
            raw_dir="data/raw/",
            processed_dir="data/processed/",
            genre_threshold=100,
            top_n_artists=10
        )
    
    @pytest.fixture
    def sample_tags_df(self):
        """Create sample tags DataFrame."""
        return pd.DataFrame({
            'tagID': [1, 2, 3, 4, 5],
            'tagValue': ['rock', 'pop', 'jazz', 'metal', 'indie']
        })
    
    @pytest.fixture
    def sample_user_tags_df(self):
        """Create sample user tags DataFrame."""
        return pd.DataFrame({
            'userID': [1, 1, 2, 2, 3],
            'artistID': [101, 102, 101, 103, 104],
            'tagID': [1, 2, 1, 3, 4]
        })
    
    @pytest.fixture
    def sample_artist_tags_df(self):
        """Create sample artist tags DataFrame."""
        return pd.DataFrame({
            'artist_id': [101, 102, 103, 104],
            'tags': [
                ['rock', 'alternative'],
                ['pop', 'dance'],
                ['jazz', 'blues'],
                ['metal', 'rock']
            ]
        })
    
    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly."""
        assert processor.raw_dir == "data/raw/"
        assert processor.processed_dir == "data/processed/"
        assert processor.genre_threshold == 100
        assert processor.top_n_artists == 10
    
    def test_extract_genres(self, processor, sample_tags_df, sample_user_tags_df):
        """Test genre extraction from tags."""
        result = processor.extract_genres(sample_tags_df, sample_user_tags_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'artist_id' in result.columns
        assert 'tags' in result.columns
        assert len(result) > 0
    
    def test_filter_common_genres(self, processor, sample_artist_tags_df):
        """Test filtering of common genres."""
        # Modify processor threshold for test
        processor.genre_threshold = 1
        
        common_genres = processor.filter_common_genres(sample_artist_tags_df)
        
        assert isinstance(common_genres, list)
        assert len(common_genres) > 0
        assert 'rock' in common_genres
    
    def test_filter_common_genres_empty(self, processor):
        """Test handling of empty artist tags."""
        empty_df = pd.DataFrame({'artist_id': [], 'tags': []})
        
        common_genres = processor.filter_common_genres(empty_df)
        
        assert isinstance(common_genres, list)
        assert len(common_genres) == 0
    
    def test_process_user_genres_exception_handling(self, processor):
        """Test exception handling in process_user_genres."""
        # Invalid dataframe should raise an error
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(Exception):
            processor.process_user_genres(invalid_df, invalid_df, [])
    
    def test_str_representation(self, processor):
        """Test __str__ method."""
        str_repr = str(processor)
        assert "DataProcessor" in str_repr
        assert "raw_dir" in str_repr
    
    def test_inheritance(self, processor):
        """Test that DataProcessor inherits from DataLoader."""
        from src.data_loader import DataLoader
        assert isinstance(processor, DataLoader)
