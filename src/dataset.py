import pandas as pd

class MusicDataset:
    """Represents a music dataset and handles loading operations."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Loads the dataset from a CSV file."""
        try:
            self.data = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file not found.")
    
    def get_features_and_labels(self):
        """Returns features and labels."""
        if self.data is None:
            raise ValueError("Data not loaded.")
        x = self.data[['tempo', 'energy', 'danceability']]
        y = self.data['preference']
        return x, y
        
    def __str__(self):
        return f"MusicDataset with {len(self.data)} rows"
