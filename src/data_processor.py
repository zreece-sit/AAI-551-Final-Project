""" Data processing module for music preference prediction. Contains DataProcessor class which inherits from DataLoader and adds data processing capabilities. """

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from typing import Tuple, List, Dict
from collections import Counter
from .data_loader import DataLoader
import os


class DataProcessor(DataLoader):
    """ Processes raw music data into training-ready format.    
    Inherits from DataLoader to access data loading functionality. """
    
    def __init__(self, raw_dir: str = "data/raw/",  # directory containing raw data
                 processed_dir: str = "data/processed/",  # directory for processed data
                 genre_threshold: int = 1000,  # minimum tag count to be considered a genre
                 top_n_artists: int = 20):  # number of top artists per user to consider
        """ Initializes DataProcessor """
                     
        super().__init__(raw_dir)
        self.processed_dir = processed_dir
        self.genre_threshold = genre_threshold
        self.top_n_artists = top_n_artists
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        
        # creates processed directory
        import os
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def extract_genres(self, tags_df: pd.DataFrame, 
                      user_tags_df: pd.DataFrame) -> pd.DataFrame:
        """ Extracts genre information from tags and merge with artists """
        
        artist_tags = user_tags_df.merge(tags_df, on='tagID', how='left')  # merges tags with user tagging data
        
        # groups tags by artist
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
        """ Identifies commonly occurring genre tags using generator expression """
        # flattens all tags using generator expression
        all_tags = (tag for tags in artist_tags_df['tags'] 
                   if isinstance(tags, list) for tag in tags)
                
        tag_counts = Counter(all_tags)  # counts frequencies
        
        # filters using lambda and filter
        common_genres = list(filter(
            lambda item: item[1] >= self.genre_threshold,
            tag_counts.items()
        ))
        
        genre_list = [tag for tag, _ in common_genres]
        
        print(f"Found {len(genre_list)} common genres "
              f"(threshold: {self.genre_threshold})")
        
        return genre_list  # returns list of common genre tags
    
    def process_user_genres(self, user_artists_df: pd.DataFrame,
                           artist_tags_df: pd.DataFrame,
                           common_genres: List[str]) -> pd.DataFrame:
        """ Creates user-genre mappings from listening data """
        # renames columns for consistency
        user_artists_df = user_artists_df.copy()  # user listening data
        user_artists_df.rename(columns={'userID': 'user_id', 'artistID': 'artist_id', 'weight': 'play_count'}, inplace=True)
        
        # gets top N artists per user using list comprehension
        top_artists = (
            user_artists_df
            .sort_values(['user_id', 'play_count'], ascending=[True, False])
            .groupby('user_id')
            .head(self.top_n_artists)
        )
        
        # merges with artist tags
        user_genres = top_artists.merge(artist_tags_df, on='artist_id', how='left')
        
        # filters to common genres using lambda
        user_genres['filtered_tags'] = user_genres['tags'].apply(
            lambda tags: [t for t in (tags if isinstance(tags, list) else []) 
                         if t in common_genres]
        )
        
        user_genres = user_genres[user_genres['filtered_tags'].map(len) > 0]  # removes users with no genres
        
        # aggregates genres per user
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
        """ Prepares feature matrix X and label matrix Y """
        # fixes demographics ID column
        demographics_df = demographics_df.copy()  # user demographic information
        
        # handles different possible ID column names
        id_cols = ['userid', 'userID', '#id', 'user_id']
        id_col = next((col for col in id_cols if col in demographics_df.columns), None)
        
        if id_col is None:
            raise ValueError(f"No valid ID column found in demographics. "
                           f"Columns: {demographics_df.columns.tolist()}")
        
        demographics_df.rename(columns={id_col: 'user_id'}, inplace=True)
        
        # removes prefix if present
        if demographics_df['user_id'].dtype == 'object':
            demographics_df['user_id'] = (
                demographics_df['user_id']
                .str.replace('user_', '', regex=False)
                .astype(int)
            )
        
        user_genre_df['user_id'] = user_genre_df['user_id'].astype(int)  # User genre preferences
        
        # merges dataframes
        final_df = user_genre_df.merge(demographics_df, on='user_id', how='inner')
        
        print(f"Final dataset shape: {final_df.shape}")
        
        # creates multi-label targets
        Y = self.mlb.fit_transform(final_df['filtered_tags'])
        genre_classes = self.mlb.classes_  # list of genre class names
        
        # prepares features
        feature_cols = [c for c in final_df.columns 
                       if c not in ['user_id', 'filtered_tags']]
        
        X_df = final_df[feature_cols].copy()
        
        # separates numeric and categorical
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # processes features
        X_num = X_df[num_cols].fillna(0) if num_cols else pd.DataFrame()
        X_cat = pd.get_dummies(X_df[cat_cols].astype(str), 
                               drop_first=True) if cat_cols else pd.DataFrame()
        
        # scales numeric features
        if not X_num.empty:
            X_num_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        else:
            X_num_scaled = X_num
        
        # combines features
        X = pd.concat([X_num_scaled, X_cat], axis=1).fillna(0).values.astype(np.float32)
        
        # stores column info for later use
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.X_cat_columns = list(X_cat.columns)
        
        return X, Y, list(genre_classes)
    
    def save_processed_data(self, X: np.ndarray, Y: np.ndarray, 
                           genre_classes: List[str]) -> None:
        """ Saves processed data and artifacts """
        import joblib
        
        # saves numpy arrays
        np.save(os.path.join(self.processed_dir, 'X.npy'), X)
        np.save(os.path.join(self.processed_dir, 'Y.npy'), Y)
        
        # saves artifacts
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
    # example
    processor = DataProcessor()
    print(processor)
