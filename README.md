# Music Preference Predictor

## Team Members
- Nicholas Obiso (nobiso@stevens.edu)
- Zachary Reece (zreece@stevens.edu)
- Rembrandt Ryan (rryan3@stevens.edu)

## Problem Description
This project predicts a user's music genre preferences based on demographic information (age, gender, country, registration date). Using the Last.fm dataset, we train a deep learning model to perform multi-label classification, predicting the top genres a user is likely to enjoy.

**Real-World Application**: Music streaming services can use this to recommend genres to new users before they've developed a listening history, improving the cold-start problem in recommendation systems.

## Dataset
- **Source**: Last.fm 1K Users Dataset (HetRec 2011)
- **URL**: http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
- **Description**: Contains user demographics, artist listening data, and genre tags

## Program Structure

### Modules (src/)
1. **data_loader.py**: `DataLoader` class - Downloads and loads raw data
2. **data_processor.py**: `DataProcessor` class - Processes and prepares data for training
3. **model_trainer.py**: `ModelTrainer` class - Builds and trains the neural network
4. **predictor.py**: `MusicPredictor` class - Makes predictions for new users
5. **visualizer.py**: `Visualizer` class - Creates charts and visualizations

### Main Notebook
- **main.ipynb**: Jupyter notebook that orchestrates the entire pipeline

## How to Use

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/music-preference-predictor.git
cd music-preference-predictor

# Install dependencies
pip install -r requirements.txt
```

### Running the Program
1. Open `main.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells in order
3. The program will:
   - Download the Last.fm dataset
   - Process user demographics and genre data
   - Train a neural network model
   - Evaluate performance
   - Make predictions for sample users

### Making Predictions for New Users
```python
from src.predictor import MusicPredictor

predictor = MusicPredictor.load_model('models/multilabel_model.h5')
new_user = {'age': 23, 'gender': 'm', 'country': 'US', 'registered': 2010}
predictions = predictor.predict(new_user, top_k=5)
```

## Requirements Met

### Part 1 Requirements
- Two classes with inheritance/composition relationships
- Multiple well-defined functions
- Advanced libraries: TensorFlow/Keras, Pandas, NumPy, Scikit-learn
- Exception handling with try-except blocks
- Pytest test cases
- Data I/O (file reading/writing)
- Loops: for, while, if statements
- Docstrings and comments

### Part 2 Requirements (4+ of 8)
- Lambda functions & filter
- List comprehension
- Built-in modules (os, urllib)
- Mutable/immutable objects
- Operator overloading (__str__, __repr__)
- Generator expressions
- __name__ == "__main__"

## Dependencies
- Python 3.12+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pytest

## License
MIT License
