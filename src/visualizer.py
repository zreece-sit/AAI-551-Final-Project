# =============================================================================
# visualizer.py - Visualization module
# =============================================================================

import matplotlib.pyplot as plt


class Visualizer:
    """
    Creates visualizations for music preference predictions.
    
    Attributes:
        predictor: MusicPredictor instance for making predictions
    """
    
    def __init__(self, predictor: MusicPredictor):
        """
        Initialize Visualizer with a predictor.
        
        Args:
            predictor: MusicPredictor instance
        """
        self.predictor = predictor
    
    def plot_top_k_predictions(self, user_dict: Dict, k: int = 5,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot horizontal bar chart of top-K predictions.
        
        Args:
            user_dict: User demographics dictionary
            k: Number of top genres to show
            figsize: Figure size (width, height)
        """
        # Get predictions
        predictions = self.predictor.predict(user_dict, top_k=k, threshold=0.0)
        
        # Extract genres and probabilities
        genres = [genre for genre, _ in predictions]
        probs = [prob for _, prob in predictions]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(genres[::-1], probs[::-1], color='skyblue', edgecolor='navy')
        plt.xlim(0, 1)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title(f'Top-{k} Genre Predictions for User', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add probability labels
        for i, (genre, prob) in enumerate(zip(genres[::-1], probs[::-1])):
            plt.text(prob + 0.02, i, f'{prob:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_genres(self, user_dict: Dict,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot probabilities for all genres.
        
        Args:
            user_dict: User demographics dictionary
            figsize: Figure size
        """
        # Get all probabilities
        all_probs = self.predictor.predict_all_probs(user_dict)
        
        # Sort by probability
        sorted_items = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        genres = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
        
        # Create plot
        plt.figure(figsize=figsize)
        colors = ['green' if p > 0.5 else 'orange' if p > 0.3 else 'red' 
                 for p in probs]
        
        plt.barh(range(len(genres)), probs, color=colors, alpha=0.7)
        plt.yticks(range(len(genres)), genres, fontsize=8)
        plt.xlabel('Probability', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title('All Genre Probabilities', fontsize=14, fontweight='bold')
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='High Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: tf.keras.callbacks.History,
                             figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Keras History object
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss Over Time')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy Over Time')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation."""
        return f"Visualizer(predictor={self.predictor})"


if __name__ == "__main__":
    print("Predictor and Visualizer modules loaded")
predictor_visualizer.py
Displaying predictor_visualizer.py.
