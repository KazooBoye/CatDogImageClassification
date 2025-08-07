import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
from DataPreprocess import DataPreprocessor
from PIL import Image
import os

# Configure GPU settings
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
            
            # Set the GPU as the logical device
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Logical GPUs: {len(logical_gpus)}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")
    
    # Print current device placement
    print(f"Default device: {tf.config.get_visible_devices()}")
    return len(gpus) > 0


class ImprovedCNNClassifier:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_anti_overfitting_cnn(self):
        """Create a CNN with balanced regularization"""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = models.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),  # Reduced dropout
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),  # Reduced dropout
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),   # Moderate dropout
                
                # Global Average Pooling
                layers.GlobalAveragePooling2D(),
                
                # Dense layers with moderate regularization
                layers.Dense(128, activation='relu'),  # Increased capacity
                layers.BatchNormalization(),
                layers.Dropout(0.5),   # Moderate dropout
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
        
        return model
    
    def build_model(self, learning_rate=0.001):  # Back to standard learning rate
        """Build and compile the model"""
        self.model = self.create_anti_overfitting_cnn()
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall'],
            jit_compile=False  # Disable XLA to avoid cuDNN issues
        )
        
        print("Anti-Overfitting CNN Model created")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Calculate trainable parameters
        trainable_params = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print which device the model is on
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Model will run on: {device}")
        
        return self.model
    
    def create_callbacks(self, model_name='improved_cnn_model'):
        """Create training callbacks with more reasonable patience"""
        callbacks_list = [
            # Save best model based on validation accuracy (more stable than loss)
            callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Less aggressive learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=5,  # More patience
                min_lr=1e-7,
                verbose=1,
                cooldown=2
            ),
            
            # Much more patient early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Much more patience
                restore_best_weights=True,
                verbose=1,
                min_delta=0.01  # Larger threshold
            ),
            
            # CSV logger
            callbacks.CSVLogger(f'../TrainingLogs/{model_name}_training_log.csv')
        ]
        
        return callbacks_list
    
    def train(self, train_generator, val_generator, epochs=25, model_name='improved_cnn_model'):
        """Train the model with overfitting prevention"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create callbacks
        callbacks_list = self.create_callbacks(model_name)
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Train model
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks_list,
                verbose=1
            )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history with overfitting analysis"""
        if self.history is None:
            print("No training history found. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy with gap analysis
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy (Check for Overfitting)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add gap analysis
        max_train_acc = max(self.history.history['accuracy'])
        max_val_acc = max(self.history.history['val_accuracy'])
        gap = max_train_acc - max_val_acc
        axes[0, 0].text(0.02, 0.98, f'Max Gap: {gap:.3f}', transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Loss with divergence analysis
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss (Lower is Better)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print overfitting analysis
        print("\n=== OVERFITTING ANALYSIS ===")
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_gap = final_train_acc - final_val_acc
        
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("Still some overfitting (gap > 0.1)")
        elif final_gap > 0.05:
            print("Moderate overfitting (gap 0.05-0.1)")
        else:
            print("Good generalization (gap < 0.05)")
    
    def evaluate_model(self, test_generator):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Evaluating model on test data...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_generator,
            steps=len(test_generator),
            verbose=1
        )
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score
        }

def main():
    # Configure GPU before doing anything else
    print("=== GPU Configuration ===")
    gpu_available = configure_gpu()
    
    # Use moderate batch size
    batch_size = 32 if gpu_available else 8
    print(f"Using batch size: {batch_size}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path="../Dataset", target_size=(224, 224))
    
    # Create data generators
    print("\n=== Creating Data Generators ===")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        '../organized_dataset', 
        batch_size=batch_size
    )
    
    # Create and train improved CNN model
    print("\n=== Training Balanced CNN ===")
    classifier = ImprovedCNNClassifier()
    classifier.build_model()
    
    # Print model summary
    print("\nModel Architecture:")
    classifier.model.summary()
    
    # Train model with more epochs
    history = classifier.train(
        train_gen, val_gen, 
        epochs=50,  # More epochs
        model_name='improved_cnn_cat_dog_model'
    )
    
    # Plot training history with overfitting analysis
    classifier.plot_training_history()
    
    # Evaluate model
    test_results = classifier.evaluate_model(test_gen)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    print(f"Model saved as: improved_cnn_cat_dog_model_best.h5")
    
    # Compare with validation accuracy to check generalization
    final_val_acc = history.history['val_accuracy'][-1]
    generalization_gap = abs(test_results['accuracy'] - final_val_acc)
    
    print(f"\nGeneralization Check:")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Generalization Gap: {generalization_gap:.4f}")
    
    if generalization_gap < 0.02:
        print("Excellent generalization!")
    elif generalization_gap < 0.05:
        print("Good generalization")
    else:
        print("Still some generalization issues")

if __name__ == "__main__":
    main()
