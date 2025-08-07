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

# Configure mixed precision for better GPU performance
def enable_mixed_precision():
    """Enable mixed precision training for better GPU performance"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled (float16)")
        return True
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
        return False

class CatDogClassifier:
    def __init__(self, input_shape=(224, 224, 1), use_mixed_precision=False):  # Changed to 1 channel for grayscale
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.use_mixed_precision = use_mixed_precision
        
    def create_simple_nn_model(self):
        """Create a simple neural network with fully connected layers"""
        # Use device scope to ensure model is created on GPU
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = models.Sequential([
                # Flatten the input images to 1D
                layers.Flatten(input_shape=self.input_shape),
                
                # First hidden layer
                # layers.Dense(512, activation='relu'),
                # layers.BatchNormalization(),
                # layers.Dropout(0.5),
                
                # Second hidden layer
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                # Third hidden layer
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Fourth hidden layer
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                # Output layer for binary classification
                # Use float32 for final layer if using mixed precision
                layers.Dense(1, activation='sigmoid', dtype='float32' if self.use_mixed_precision else None)
            ])
        
        return model
    
    def build_model(self):
        """Build and compile the model"""
        self.model = self.create_simple_nn_model()
        
        # Compile model with appropriate optimizer for mixed precision
        optimizer = optimizers.Adam(learning_rate=0.001)
        if self.use_mixed_precision:
            # Wrap optimizer for mixed precision
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Simple Neural Network Model created")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Print which device the model is on
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Model will run on: {device}")
        
        return self.model
    
    def create_callbacks(self, model_name='simple_cat_dog_model'):
        """Create training callbacks"""
        callbacks_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(f'../TrainingLogs/{model_name}_training_log.csv')
        ]
        
        return callbacks_list
    
    def train(self, train_generator, val_generator, epochs=60, model_name='simple_cat_dog_model'):
        """Train the model"""
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
        
        # Monitor GPU usage during training
        def print_gpu_usage():
            if tf.config.list_physical_devices('GPU'):
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                           '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        memory_used, memory_total, gpu_util = result.stdout.strip().split(', ')
                        print(f"GPU Memory: {memory_used}MB / {memory_total}MB, GPU Utilization: {gpu_util}%")
                except:
                    pass
        
        print("GPU status before training:")
        print_gpu_usage()
        
        # Train model with GPU monitoring
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
        
        print("GPU status after training:")
        print_gpu_usage()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history found. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
        
        print(f"\nTest Results:")
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
    
    # Enable mixed precision if GPU is available
    mixed_precision_enabled = False
    if gpu_available:
        mixed_precision_enabled = enable_mixed_precision()
    
    # Initialize preprocessor and create data generators
    preprocessor = DataPreprocessor(data_path="../Dataset", target_size=(224, 224))
    
    # Create data generators with GPU-optimized settings
    print("\n=== Creating Data Generators ===")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        '../organized_dataset', 
        batch_size=64 if gpu_available else 8  # Larger batch size for GPU
    )
    
    # Create and train simple neural network
    print("\n=== Training Simple Neural Network ===")
    classifier = CatDogClassifier(use_mixed_precision=mixed_precision_enabled)
    classifier.build_model()
    classifier.model.summary()
    
    # Train model
    history = classifier.train(
        train_gen, val_gen, 
        epochs=100,  # More epochs since simple NN may need more training
        model_name='simple_cat_dog_model'
    )
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    results = classifier.evaluate_model(test_gen)
    
    print(f"\nFinal Results Summary:")
    print(f"Best Test Accuracy: {results['accuracy']:.4f}")
    print(f"Model saved as: simple_cat_dog_model_best.h5")
    print(f"GPU was {'used' if gpu_available else 'not available'}")
    print(f"Mixed precision was {'enabled' if mixed_precision_enabled else 'disabled'}")

if __name__ == "__main__":
    main()