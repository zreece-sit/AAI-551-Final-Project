import pandas as pd
from typing import Dict, Tuple


class DataProc:
    """
    Processes raw Last.fm dataset into cleaned and structured DataFrames.
    """

    def __init__(self, user_artists: pd.DataFrame, demographics: pd.DataFrame,
                 artists: pd.DataFrame, user_tags: pd.DataFrame, tags: pd.DataFrame):
        """
        Initialize DataProc with raw dataframes.
            user_artists: User listening data
            demographics: User demographic information
            artists: Artist information
            user_tags: User tagging data
            tags: Tag definitions
        """
        self.user_artists = user_artists
        self.demographics = demographics
        self.artists = artists
        self.user_tags = user_tags
        self.tags = tags

    def clean_demographics(self) -> pd.DataFrame:
        """
        Clean demographics data 
        Returns:
            pd.DataFrame: Cleaned demographics
        """
        df = self.demographics.copy()
        df = df.dropna(subset=["age", "gender", "country"])
        df["gender"] = df["gender"].str.lower().map({"m": "male", "f": "female"})
        df["country"] = df["country"].str.strip().str.title()
        return df

    def clean_artists(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Cleaned artists
        """
        df = self.artists.copy()
        df = df.drop_duplicates(subset=["id"])
        df["name"] = df["name"].str.strip().str.title()
        return df

    def merge_user_artist_data(self) -> pd.DataFrame:
        """
        Merge user listening data with artist metadata.
        """
        df = self.user_artists.merge(self.artists, left_on="artistID", right_on="id", how="left")
        return df

    def process_tags(self) -> pd.DataFrame:
        """
        Merge user tags with tag definitions.
        """
        df = self.user_tags.merge(self.tags, left_on="tagID", right_on="id", how="left")
        return df

    def build_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Build a dictionary of cleaned datasets.
        """
        return {
            "demographics": self.clean_demographics(),
            "artists": self.clean_artists(),
            "user_artist_data": self.merge_user_artist_data(),
            "tags": self.process_tags()
        }


if __name__ == "__main__":
    from data_loader import DataLoader, validate_dataframes

    loader = DataLoader()
    loader.download_dataset()
    loader.extract_dataset()
    user_artists, demographics, artists, user_tags, tags = loader.load_raw_data()

    if validate_dataframes(user_artists, demographics, artists, user_tags, tags):
        processor = DataProc(user_artists, demographics, artists, user_tags, tags)
        datasets = processor.build_dataset()
        for name, df in datasets.items():
            print(f"{name}: {df.shape}")
    else:
        print("Invalid raw dataframes.")
