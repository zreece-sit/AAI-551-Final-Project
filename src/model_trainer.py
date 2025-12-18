""" Model training module for music genre preference prediction. Contains the ModelTrainer class for building and training neural network models. """

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict


class ModelTrainer:
    """ Handles neural network model creation and training """
    
    def __init__(self, input_dim: int, output_dim: int):
        """ Initializes ModelTrainer """
        self.input_dim = input_dim  # number of input features
        self.output_dim = output_dim  # number of output classes
        self.model = None  # neural network model
        self.history = None  # training history object
    
    def build_model(self, hidden_layers: list = [256, 128, 64],
                   dropout_rates: list = [0.3, 0.2, 0.1]) -> None:
        """ Builds multi-layer neural network for multi-label classification """
        
        inp = layers.Input(shape=(self.input_dim,), name='input_layer')  # Input layer
        x = inp
        
        # hidden layers with dropout using enumerate
        for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)): # lists of hidden layer sizes & dropout rates for each layer
            x = layers.Dense(units, activation='relu', name=f'hidden_{i+1}')(x)
            x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        out = layers.Dense(self.output_dim, activation='sigmoid', name='output_layer')(x) # Output layer (sigmoid for multi-label)
        
        # creates and compile model
        self.model = models.Model(inputs=inp, outputs=out, name='genre_predictor')
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        print(f"Input dimension: {self.input_dim}")
        print(f"Output dimension: {self.output_dim}")

    
    def train(self, X: np.ndarray, Y: np.ndarray,
             test_size: float = 0.3,  # proportion for test set
             val_size: float = 0.5,  # proportion of remaining for validation
             epochs: int = 100,  # number of training epochs
             batch_size: int = 128,  # batch size for training
             model_save_path: str = 'models/multilabel_model.h5',  # path to save best model
             random_state: int = 42) -> Dict:  # random seed for reproducibility
    """ Trains the model with train/validation/test split """
                 
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # splits data: train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        
        # splits temp: val + test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
        
        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        
        # ensures float32 dtype
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # creates callbacks
        
        # trains model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            verbose=2
        )
        
        print("\nTraining complete!")

        # returns dictionary with train/val/test data splits
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:  # test features & labels
        """ Evaluate model on test set """
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # gets predictions
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        
        # calculates metrics
        from sklearn.metrics import hamming_loss, f1_score
        
        metrics = {
            'hamming_loss': hamming_loss(y_test, y_pred),
            'micro_f1': f1_score(y_test, y_pred, average='micro'),
            'macro_f1': f1_score(y_test, y_pred, average='macro')
        }
        
        print("\nTest Set Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics  # returns dictionary of evaluation metrics
    
    def summary(self) -> None:
        """ Prints model summary """
        
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()
    
    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """ Loads saved model from file """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return tf.keras.models.load_model(model_path)
    
    def __str__(self) -> str:
        """ String representation """
        return (f"ModelTrainer(input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"model_built={self.model is not None})")
    
    def __repr__(self) -> str:
        return self.__str__()


def calculate_top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    """ Calculates top-k accuracy for multi-label classification """
    hits = 0
    
    i = 0  # uses while loop for iteration
    while i < y_true.shape[0]:
        
        topk_idx = np.argsort(y_prob[i])[-k:][::-1] ## Get top-k predicted indices
        true_idx = set(np.where(y_true[i] == 1)[0])  # gets true label indices
        
        if len(true_idx.intersection(topk_idx)) > 0:  # checks if any top-k prediction is correct
            hits += 1
        
        i += 1
    
    return hits / y_true.shape[0]


if __name__ == "__main__":
    print("ModelTrainer module loaded")
    
    # creates example data for testing
    X_dummy = np.random.rand(100, 20).astype(np.float32)
    Y_dummy = np.random.randint(0, 2, size=(100, 10)).astype(np.float32)
    
    trainer = ModelTrainer(input_dim=20, output_dim=10)
    trainer.build_model()
    print(trainer)
