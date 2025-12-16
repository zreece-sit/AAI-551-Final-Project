"""
Data loading module for Last.fm dataset.
This module handles downloading and initial loading of the Last.fm dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
from typing import Tuple, Optional


class DataLoader:
    """
    Handles downloading and loading of Last.fm music dataset.
    Attributes:
        raw_dir (str): Directory path for raw data storage
        url (str): URL to download the dataset from
    """
    
    def __init__(self, raw_dir: str = "data/raw/"):
        """
        Initialize DataLoader with directory path.
        Args:
            raw_dir (str): Directory to store raw data files
        """
        self.raw_dir = raw_dir
        self.url = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def download_dataset(self) -> bool:
        """
        Download the Last.fm dataset if not already present.
        Returns:
            bool: True if download successful or file exists, False otherwise
        Raises:
            urllib.error.URLError: If download fails
        """
        zip_path = os.path.join(self.raw_dir, "lastfm.zip")
        
        # Check if file already exists
        if os.path.exists(zip_path):
            print("Dataset already downloaded.")
            return True
        try:
            print("Downloading Last.fm dataset...")
            urllib.request.urlretrieve(self.url, zip_path)
            print("Download complete!")
            return True
        except urllib.error.URLError as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def extract_dataset(self) -> None:
        """
        Extract the downloaded zip file.
        Raises:
            FileNotFoundError: If zip file doesn't exist
            zipfile.BadZipFile: If zip file is corrupted
        """
        zip_path = os.path.join(self.raw_dir, "lastfm.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found at {zip_path}")
        try:
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.raw_dir)
            print("Extraction complete!")
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip file: {e}")
            raise
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, ...]:
        """
        Load all raw data files into pandas DataFrames.
        Returns:
            Tuple containing:
                - user_artists (pd.DataFrame): User listening data
                - demographics (pd.DataFrame): User demographic information
                - artists (pd.DataFrame): Artist information
                - user_tags (pd.DataFrame): User tagging data
                - tags (pd.DataFrame): Tag definitions
        Raises:
            FileNotFoundError: If required data files are missing
        """
        try:
            # Define file paths
            files = {
                'user_artists': 'user_artists.dat',
                'demographics': 'user_profiles.tsv',
                'artists': 'artists.dat',
                'user_tags': 'user_taggedartists.dat',
                'tags': 'tags.dat'
            }
            # Load each file
            data = {}
            for key, filename in files.items():
                filepath = os.path.join(self.raw_dir, filename)
                
                # Handle different file formats
                sep = '\t' if filename.endswith(('.dat', '.tsv')) else ','
                
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Required file not found: {filepath}")
                
                data[key] = pd.read_csv(filepath, sep=sep, encoding='ISO-8859-1')
                print(f"Loaded {key}: {data[key].shape}")
            
            return (
                data['user_artists'],
                data['demographics'],
                data['artists'],
                data['user_tags'],
                data['tags']
            )
        except Exception as e:
            print(f"Error loading raw data: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of DataLoader."""
        return f"DataLoader(raw_dir='{self.raw_dir}')"
    
    def __repr__(self) -> str:
        """Official string representation of DataLoader."""
        return f"DataLoader(raw_dir='{self.raw_dir}', url='{self.url}')"
        
def validate_dataframes(*dfs: pd.DataFrame) -> bool:
    """
    Validate that dataframes are not empty.
    Args:
        *dfs: Variable number of DataFrames to validate
    Returns:
        bool: True if all dataframes are valid, False otherwise
    """
    return all(not df.empty for df in dfs)

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    loader.download_dataset()
    loader.extract_dataset()
    data = loader.load_raw_data()
    print(f"Successfully loaded {len(data)} dataframes")
