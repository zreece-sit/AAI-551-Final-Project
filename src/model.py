from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class GenrePredictor:
    """
    Builds and trains a deep learning model to predict music genres.
    """

    def __init__(self, dataset):
        self.dataset = dataset  # composition
        self.model = None

    def build_model(self, input_dim, output_dim):
        """Creates and compiles the neural network."""
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X, y, epochs=1):
        """Trains the model with exception handling."""
        try:
            self.model.fit(X, y, epochs=epochs)
        except ValueError:
            print("Training failed due to invalid input data.")
