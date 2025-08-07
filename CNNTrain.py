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

class CatDogCNNClassifier:
    def __init__(self, input_shape=(224, 224, 3), use_mixed_precision=False, model_type='cnn'):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.use_mixed_precision = use_mixed_precision
        self.model_type = model_type
        
    def create_cnn_model(self):
        """Create a CNN model optimized for image classification"""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = models.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fourth Convolutional Block
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Global Average Pooling (better than Flatten for CNNs)
                layers.GlobalAveragePooling2D(),
                
                # Dense layers
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                # Output layer
                layers.Dense(1, activation='sigmoid', dtype='float32' if self.use_mixed_precision else None)
            ])
        
        return model
    
    def create_lightweight_cnn_model(self):
        """Create a lightweight CNN model with better regularization"""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = models.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),  # Increased dropout
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.35),  # Increased dropout
                
                # Third Convolutional Block
                layers.Conv2D(96, (3, 3), activation='relu', padding='same',  # Reduced filters
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.4),  # Increased dropout
                
                # Global Average Pooling
                layers.GlobalAveragePooling2D(),
                
                # Simpler dense layers to reduce overfitting
                layers.Dense(64, activation='relu',  # Reduced from 128
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.6),  # Higher dropout before output
                
                # Output layer
                layers.Dense(1, activation='sigmoid', dtype='float32' if self.use_mixed_precision else None)
            ])
        
        return model
    
    def create_resnet_style_model(self):
        """Create a ResNet-inspired architecture"""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            
            def residual_block(x, filters, kernel_size=3, stride=1):
                """Create a residual block"""
                shortcut = x
                
                # First conv layer
                x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                
                # Second conv layer
                x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
                x = layers.BatchNormalization()(x)
                
                # Adjust shortcut if needed
                if stride != 1 or shortcut.shape[-1] != filters:
                    shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                    shortcut = layers.BatchNormalization()(shortcut)
                
                # Add shortcut
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)
                
                return x
            
            # Input layer
            inputs = layers.Input(shape=self.input_shape)
            
            # Initial conv
            x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            
            # Residual blocks
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            
            x = residual_block(x, 128, stride=2)
            x = residual_block(x, 128)
            
            x = residual_block(x, 256, stride=2)
            x = residual_block(x, 256)
            
            # Global average pooling and classification
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation='sigmoid', 
                                  dtype='float32' if self.use_mixed_precision else None)(x)
            
            model = models.Model(inputs, outputs)
        
        return model
    
    def create_model(self):
        """Create model based on specified type"""
        if self.model_type == 'cnn':
            return self.create_cnn_model()
        elif self.model_type == 'lightweight_cnn':
            return self.create_lightweight_cnn_model()
        elif self.model_type == 'resnet_style':
            return self.create_resnet_style_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def build_model(self, learning_rate=0.0005):  # Reduced from 0.001
        """Build and compile the model"""
        self.model = self.create_model()
        
        # Compile model with XLA optimization
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall'],
            jit_compile=True  # Enable XLA compilation for faster execution
        )
        
        print(f"{self.model_type.upper()} Model created with XLA optimization")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Calculate trainable parameters
        trainable_params = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print which device the model is on
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Model will run on: {device}")
        
        return self.model
    
    def create_callbacks(self, model_name='cnn_cat_dog_model'):
        """Create training callbacks with stronger overfitting prevention"""
        callbacks_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # More aggressive learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction
                patience=2,  # Faster response
                min_lr=1e-8,
                verbose=1
            ),
            
            # Earlier stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',  # Monitor validation loss instead of accuracy
                patience=5,  # Much more aggressive
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Only stop if no improvement
            ),
            
            # CSV logger
            callbacks.CSVLogger(f'{model_name}_training_log.csv')
        ]
        
        return callbacks_list
    
    def train(self, train_generator, val_generator, epochs=50, model_name='cnn_cat_dog_model'):
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
        plt.savefig(f'{self.model_type}_training_history.png', dpi=300, bbox_inches='tight')
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
    
    def predict_single_image(self, image_path):
        """Predict a single image"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(self.input_shape[:2])
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_array)
        probability = prediction[0][0]
        
        # Interpret result
        class_name = "Dog" if probability > 0.5 else "Cat"
        confidence = probability if probability > 0.5 else 1 - probability
        
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        
        return class_name, confidence

def find_optimal_batch_size(input_shape=(224, 224, 3)):
    """Find the largest batch size that fits in GPU memory"""
    if not tf.config.list_physical_devices('GPU'):
        return 16
    
    # Test batch sizes for 6GB GPU
    test_sizes = [16, 32, 48, 64, 80, 96, 128]
    max_working_size = 16
    
    print("Testing optimal batch size for 6GB GPU...")
    
    for batch_size in test_sizes:
        try:
            print(f"  Testing batch size: {batch_size}")
            
            # Create a test CNN model
            with tf.device('/GPU:0'):
                test_model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                # Test with dummy data
                dummy_data = tf.random.normal([batch_size] + list(input_shape))
                dummy_labels = tf.random.uniform([batch_size, 1])
                
                # Compile and try a forward pass
                test_model.compile(optimizer='adam', loss='binary_crossentropy')
                _ = test_model(dummy_data)
                
                # Try a training step
                with tf.GradientTape() as tape:
                    predictions = test_model(dummy_data)
                    loss = tf.keras.losses.binary_crossentropy(dummy_labels, predictions)
                
                gradients = tape.gradient(loss, test_model.trainable_variables)
            
            max_working_size = batch_size
            print(f"    ✓ Batch size {batch_size} works")
            
            # Clear memory
            del test_model, dummy_data, dummy_labels, predictions, loss, gradients
            tf.keras.backend.clear_session()
            
        except tf.errors.ResourceExhaustedError:
            print(f"    ✗ Batch size {batch_size} - Out of memory")
            break
        except Exception as e:
            print(f"    ✗ Batch size {batch_size} - Error: {e}")
            break
    
    print(f"Optimal batch size: {max_working_size}")
    return max_working_size

def main():
    # Configure GPU before doing anything else
    print("=== GPU Configuration ===")
    gpu_available = configure_gpu()
    
    # Enable mixed precision if GPU is available
    mixed_precision_enabled = False
    if gpu_available:
        mixed_precision_enabled = enable_mixed_precision()
    
    # Find optimal batch size for your 6GB GPU
    optimal_batch_size = 16  # Default fallback
    if gpu_available:
        optimal_batch_size = find_optimal_batch_size()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path="./Dataset", target_size=(224, 224))
    
    # Create data generators with optimal batch size
    print("\n=== Creating Data Generators ===")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        'organized_dataset', 
        batch_size=optimal_batch_size
    )
    
    # Train different CNN models
    models_to_train = [
        ('lightweight_cnn', 'Lightweight CNN'),
        ('cnn', 'Standard CNN'),
        ('resnet_style', 'ResNet-Style CNN'),
    ]
    
    results = {}
    
    for model_type, model_name in models_to_train:
        print(f"\n=== Training {model_name} ===")
        
        classifier = CatDogCNNClassifier(
            use_mixed_precision=mixed_precision_enabled,
            model_type=model_type
        )
        classifier.build_model()
        
        # Print model summary
        print(f"\n{model_name} Architecture:")
        classifier.model.summary()
        
        # Train model
        history = classifier.train(
            train_gen, val_gen, 
            epochs=30,  # CNNs converge faster than dense networks
            model_name=f'{model_type}_cat_dog_model'
        )
        
        # Plot training history
        classifier.plot_training_history()
        
        # Evaluate model
        test_results = classifier.evaluate_model(test_gen)
        results[model_name] = test_results
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1-Score: {test_results['f1_score']:.4f}")
        print(f"  Model saved as: {model_type}_cat_dog_model_best.h5")
        
        # Clear memory before next model
        tf.keras.backend.clear_session()
    
    # Compare results
    print(f"\n=== FINAL COMPARISON ===")
    print("Model                    | Accuracy | F1-Score | Parameters")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<24} | {result['accuracy']:.4f}   | {result['f1_score']:.4f}   |")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest performing model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")

if __name__ == "__main__":
    main()
