""" Data loading module for Last.fm dataset. Handles downloading and initial loading of the Last.fm dataset. """

import os
import urllib.request
import zipfile
import pandas as pd
from typing import Tuple, Optional


class DataLoader:
    """ Handles downloading and loading of Last.fm music dataset """
    
    def __init__(self, raw_dir: str = "data/raw/"):
        """ Initializes DataLoader with directory path """
        self.raw_dir = raw_dir  # directory path for raw data storage
        self.url = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"  # URL to download dataset from
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def download_dataset(self) -> bool:
        """ Downloads Last.fm dataset if not already present """
        
        zip_path = os.path.join(self.raw_dir, "lastfm.zip")
        
        # checks if file already exists
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
        """ Extracts downloaded zip file"""
        
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
        """ Loads all raw data files into pandas DataFrames """
        
        try:
            # defines file paths
            files = {
                'user_artists': 'user_artists.dat',  # user listening data
                'demographics': 'user_profiles.tsv',  # user demographic information
                'artists': 'artists.dat',  # artist information
                'user_tags': 'user_taggedartists.dat',  # user tagging data
                'tags': 'tags.dat'  # tag definitions
            }
            # loads each file
            data = {}
            for key, filename in files.items():
                filepath = os.path.join(self.raw_dir, filename)
                
                sep = '\t' if filename.endswith(('.dat', '.tsv')) else ',' # handles different file formats
                
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
        """String representation"""
        return f"DataLoader(raw_dir='{self.raw_dir}')"
    
    def __repr__(self) -> str:
        return f"DataLoader(raw_dir='{self.raw_dir}', url='{self.url}')"
        
def validate_dataframes(*dfs: pd.DataFrame) -> bool:
    """ Validates dataframes are not empty """

    return all(not df.empty for df in dfs)  # returns True if all dataframes are valid, False otherwise

if __name__ == "__main__":
    # example
    loader = DataLoader()
    loader.download_dataset()
    loader.extract_dataset()
    data = loader.load_raw_data()
    print(f"Successfully loaded {len(data)} dataframes")
