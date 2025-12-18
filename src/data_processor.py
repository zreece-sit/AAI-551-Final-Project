"""
Data processing module for music preference prediction.

This module contains the DataProcessor class which inherits from DataLoader
and adds data processing capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from typing import Tuple, List, Dict
from collections import Counter
from .data_loader import DataLoader
import os


class DataProcessor(DataLoader):
    """
    Processes raw music data into training-ready format.
    
    Inherits from DataLoader to access data loading functionality.
    
    Attributes:
        processed_dir (str): Directory for processed data
        genre_threshold (int): Minimum tag count to be considered a genre
        top_n_artists (int): Number of top artists per user to consider
    """
    
    def __init__(self, raw_dir: str = "data/raw/", 
                 processed_dir: str = "data/processed/",
                 genre_threshold: int = 1000,
                 top_n_artists: int = 20):
        """
        Initialize DataProcessor.
        
        Args:
            raw_dir: Directory containing raw data
            processed_dir: Directory to save processed data
            genre_threshold: Minimum occurrences for a tag to be a genre
            top_n_artists: Top N artists per user to consider
        """
        super().__init__(raw_dir)
        self.processed_dir = processed_dir
        self.genre_threshold = genre_threshold
        self.top_n_artists = top_n_artists
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        
        # Create processed directory
        import os
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def extract_genres(self, tags_df: pd.DataFrame, 
                      user_tags_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract genre information from tags and merge with artists.
        
        Args:
            tags_df: DataFrame containing tag definitions
            user_tags_df: DataFrame containing user tagging data
            
        Returns:
            pd.DataFrame: Artist-genre mapping
        """
        # Merge tags with user tagging data
        artist_tags = user_tags_df.merge(tags_df, on='tagID', how='left')
        
        # Group tags by artist
        artist_tags_grouped = (
            artist_tags.groupby('artistID')['tagValue']
            .apply(lambda x: list(set(x.dropna().tolist())))
            .reset_index()
        )
        
        artist_tags_grouped.rename(
            columns={'artistID': 'artist_id', 'tagValue': 'tags'},
            inplace=True
        )
        
        return artist_tags_grouped
    
    def filter_common_genres(self, artist_tags_df: pd.DataFrame) -> List[str]:
        """
        Identify commonly occurring genre tags using generator expression.
        
        Args:
            artist_tags_df: DataFrame with artist tags
            
        Returns:
            List of common genre tags
        """
        # Flatten all tags using generator expression
        all_tags = (tag for tags in artist_tags_df['tags'] 
                   if isinstance(tags, list) for tag in tags)
        
        # Count frequencies
        tag_counts = Counter(all_tags)
        
        # Filter using lambda and filter
        common_genres = list(filter(
            lambda item: item[1] >= self.genre_threshold,
            tag_counts.items()
        ))
        
        # Return just the tag names
        genre_list = [tag for tag, _ in common_genres]
        
        print(f"Found {len(genre_list)} common genres "
              f"(threshold: {self.genre_threshold})")
        
        return genre_list
    
    def process_user_genres(self, user_artists_df: pd.DataFrame,
                           artist_tags_df: pd.DataFrame,
                           common_genres: List[str]) -> pd.DataFrame:
        """
        Create user-genre mappings from listening data.
        
        Args:
            user_artists_df: User listening data
            artist_tags_df: Artist tag mappings
            common_genres: List of valid genre tags
            
        Returns:
            pd.DataFrame: User-genre relationships
        """
        # Rename columns for consistency
        user_artists_df = user_artists_df.copy()
        user_artists_df.rename(
            columns={'userID': 'user_id', 'artistID': 'artist_id', 
                    'weight': 'play_count'},
            inplace=True
        )
        
        # Get top N artists per user using list comprehension
        top_artists = (
            user_artists_df
            .sort_values(['user_id', 'play_count'], ascending=[True, False])
            .groupby('user_id')
            .head(self.top_n_artists)
        )
        
        # Merge with artist tags
        user_genres = top_artists.merge(artist_tags_df, on='artist_id', how='left')
        
        # Filter to common genres using lambda
        user_genres['filtered_tags'] = user_genres['tags'].apply(
            lambda tags: [t for t in (tags if isinstance(tags, list) else []) 
                         if t in common_genres]
        )
        
        # Remove users with no genres
        user_genres = user_genres[user_genres['filtered_tags'].map(len) > 0]
        
        # Aggregate genres per user
        user_genre_multi = (
            user_genres.groupby('user_id')['filtered_tags']
            .apply(lambda tag_lists: list(set([
                tag for lst in tag_lists for tag in lst
            ])))
            .reset_index()
        )
        
        print(f"Processed {len(user_genre_multi)} users with genre preferences")
        
        return user_genre_multi
    
    def prepare_features_and_labels(
        self, 
        user_genre_df: pd.DataFrame,
        demographics_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix X and label matrix Y.
        
        Args:
            user_genre_df: User genre preferences
            demographics_df: User demographic information
            
        Returns:
            Tuple of (X, Y, genre_classes) where:
                X: Feature matrix
                Y: Multi-label target matrix
                genre_classes: List of genre class names
        """
        # Fix demographics ID column
        demographics_df = demographics_df.copy()
        
        # Handle different possible ID column names
        id_cols = ['userid', 'userID', '#id', 'user_id']
        id_col = next((col for col in id_cols if col in demographics_df.columns), None)
        
        if id_col is None:
            raise ValueError(f"No valid ID column found in demographics. "
                           f"Columns: {demographics_df.columns.tolist()}")
        
        demographics_df.rename(columns={id_col: 'user_id'}, inplace=True)
        
        # Remove prefix if present
        if demographics_df['user_id'].dtype == 'object':
            demographics_df['user_id'] = (
                demographics_df['user_id']
                .str.replace('user_', '', regex=False)
                .astype(int)
            )
        
        user_genre_df['user_id'] = user_genre_df['user_id'].astype(int)
        
        # Merge dataframes
        final_df = user_genre_df.merge(demographics_df, on='user_id', how='inner')
        
        print(f"Final dataset shape: {final_df.shape}")
        
        # Create multi-label targets
        Y = self.mlb.fit_transform(final_df['filtered_tags'])
        genre_classes = self.mlb.classes_
        
        # Prepare features
        feature_cols = [c for c in final_df.columns 
                       if c not in ['user_id', 'filtered_tags']]
        
        X_df = final_df[feature_cols].copy()
        
        # Separate numeric and categorical
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Process features
        X_num = X_df[num_cols].fillna(0) if num_cols else pd.DataFrame()
        X_cat = pd.get_dummies(X_df[cat_cols].astype(str), 
                               drop_first=True) if cat_cols else pd.DataFrame()
        
        # Scale numeric features
        if not X_num.empty:
            X_num_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        else:
            X_num_scaled = X_num
        
        # Combine features
        X = pd.concat([X_num_scaled, X_cat], axis=1).fillna(0).values.astype(np.float32)
        
        # Store column info for later use
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.X_cat_columns = list(X_cat.columns)
        
        return X, Y, list(genre_classes)
    
    def save_processed_data(self, X: np.ndarray, Y: np.ndarray, 
                           genre_classes: List[str]) -> None:
        """
        Save processed data and artifacts.
        
        Args:
            X: Feature matrix
            Y: Label matrix
            genre_classes: List of genre names
        """
        import joblib
        
        # Save numpy arrays
        np.save(os.path.join(self.processed_dir, 'X.npy'), X)
        np.save(os.path.join(self.processed_dir, 'Y.npy'), Y)
        
        # Save artifacts
        artifacts_dir = 'artifacts'
        os.makedirs(artifacts_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
        joblib.dump(self.mlb, os.path.join(artifacts_dir, 'mlb.joblib'))
        joblib.dump({
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'X_cat_columns': self.X_cat_columns,
            'genre_classes': genre_classes
        }, os.path.join(artifacts_dir, 'cols_info.joblib'))
        
        print("Processed data and artifacts saved successfully!")
    
    def __str__(self) -> str:
        """String representation."""
        return (f"DataProcessor(raw_dir='{self.raw_dir}', "
                f"processed_dir='{self.processed_dir}')")


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    print(processor)
