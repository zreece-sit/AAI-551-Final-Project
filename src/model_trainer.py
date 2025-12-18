# Save this as src/improved_model_trainer.py or modify your existing one

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

class ImprovedModelTrainer:
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_improved_model(self, hidden_layers=[256, 128, 64], 
                           dropout_rates=[0.3, 0.2, 0.1],
                           use_batch_norm=True,
                           use_residual=False):
        """
        Build an improved neural network model.
        """
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = inputs
        
        # Add initial batch normalization
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        # Add hidden layers with options
        for i, (units, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            # Dense layer
            x_dense = tf.keras.layers.Dense(units, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            
            # Batch normalization
            if use_batch_norm:
                x_dense = tf.keras.layers.BatchNormalization()(x_dense)
            
            # Dropout
            x_dense = tf.keras.layers.Dropout(dropout_rate)(x_dense)
            
            # Residual connection (optional)
            if use_residual and x.shape[-1] == units:
                x = tf.keras.layers.add([x, x_dense])
            else:
                x = x_dense
        
        # Output layer with sigmoid for multilabel classification
        outputs = tf.keras.layers.Dense(self.output_dim, activation='sigmoid')(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with Adam optimizer and appropriate learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # For multilabel classification
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Changed from categorical_crossentropy
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
    
    def train_with_class_weights(self, X, y, test_size=0.3, val_size=0.5,
                                epochs=100, batch_size=64, random_state=42):
        """
        Train with class weights to handle imbalance.
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=random_state
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,  # Add class weights
            verbose=1
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def _calculate_class_weights(self, y):
        """
        Calculate class weights for imbalanced data.
        """
        n_samples = len(y)
        n_classes = y.shape[1]
        
        weights = []
        for i in range(n_classes):
            class_count = np.sum(y[:, i])
            if class_count == 0:
                weight = 0
            else:
                weight = n_samples / (n_classes * class_count)
            weights.append(min(weight, 10))  # Cap weights to avoid extremes
        
        # Convert to dictionary
        return {i: weights[i] for i in range(n_classes)}
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {}
        
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            print(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict_top_k(self, X, k=5):
        """Get top-k predictions."""
        probabilities = self.model.predict(X, verbose=0)
        top_k_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.take_along_axis(probabilities, top_k_indices, axis=1)
        
        return top_k_indices, top_k_probs
