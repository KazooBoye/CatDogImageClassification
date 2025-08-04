import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
from DataPreprocess import DataPreprocessor
from PIL import Image
import os

class CatDogClassifier:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_simple_nn_model(self):
        """Create a simple neural network with fully connected layers"""
        model = models.Sequential([
            # Flatten the input images to 1D
            layers.Flatten(input_shape=self.input_shape),
            
            # First hidden layer
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
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
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_model(self):
        """Build and compile the model"""
        self.model = self.create_simple_nn_model()
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Simple Neural Network Model created")
        print(f"Total parameters: {self.model.count_params():,}")
        
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
            callbacks.CSVLogger(f'{model_name}_training_log.csv')
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
        
        # Train model
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

def debug_dataset(dataset_path):
    """Debug organized dataset to find corrupted files without deleting them"""
    
    corrupted_files = []
    total_files = 0
    metadata_files = []
    
    print("=== DEBUGGING ORGANIZED DATASET ===")
    
    for split in ['train', 'val', 'test']:
        print(f"\nChecking {split} split:")
        for class_name in ['cats', 'dogs']:
            folder_path = os.path.join(dataset_path, split, class_name)
            if not os.path.exists(folder_path):
                print(f"  Folder not found: {folder_path}")
                continue
                
            files_in_folder = os.listdir(folder_path)
            print(f"  {class_name}: {len(files_in_folder)} files")
            
            for filename in files_in_folder:
                file_path = os.path.join(folder_path, filename)
                
                # Check for macOS metadata files
                if filename.startswith('.') or filename.startswith('._'):
                    metadata_files.append(file_path)
                    print(f"    METADATA FILE: {filename}")
                    continue
                
                # Check image files
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    total_files += 1
                    
                    try:
                        # Try to open and validate the image
                        with Image.open(file_path) as img:
                            img.load()  # Force load
                            # Check basic properties
                            if img.size[0] < 10 or img.size[1] < 10:
                                print(f"    TINY IMAGE: {filename} - size: {img.size}")
                            
                        # Try verify (this closes the file)
                        with Image.open(file_path) as img:
                            img.verify()
                        
                        # Try conversion
                        with Image.open(file_path) as img:
                            img.convert('RGB')
                            
                    except Exception as e:
                        corrupted_files.append(file_path)
                        print(f"    CORRUPTED: {filename} - Error: {str(e)}")
                        
                        # Try to get file size to debug further
                        try:
                            file_size = os.path.getsize(file_path)
                            print(f"      File size: {file_size} bytes")
                        except:
                            print(f"      Cannot get file size")
                
                else:
                    print(f"    NON-IMAGE FILE: {filename}")
    
    print(f"\n=== DEBUGGING SUMMARY ===")
    print(f"Total image files: {total_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Metadata files: {len(metadata_files)}")
    
    if corrupted_files:
        print(f"\n=== CORRUPTED FILES LIST ===")
        for i, file_path in enumerate(corrupted_files, 1):
            print(f"{i:2d}. {file_path}")
    
    if metadata_files:
        print(f"\n=== METADATA FILES LIST ===")
        for i, file_path in enumerate(metadata_files, 1):
            print(f"{i:2d}. {file_path}")
    
    return corrupted_files, metadata_files

def clean_metadata_files(dataset_path):
    """Remove macOS metadata files from organized dataset"""
    import os
    
    removed_count = 0
    
    for split in ['train', 'val', 'test']:
        for class_name in ['cats', 'dogs']:
            folder_path = os.path.join(dataset_path, split, class_name)
            if not os.path.exists(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if filename.startswith('.') or filename.startswith('._'):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Removed: {filename}")
    
    print(f"Removed {removed_count} metadata files")

def main():
    # Initialize preprocessor and create data generators
    preprocessor = DataPreprocessor(data_path="./Dataset", target_size=(224, 224))
    
    # Debug and remove metadata files from the dataset first
   
    print("Debugging organized dataset...")
    corrupted_files, metadata_files = debug_dataset('organized_dataset') 
    if metadata_files:
        print(f"\n Cleaning {len(metadata_files)} metadata files...")
        clean_metadata_files('organized_dataset')
        print("Metadata files cleaned")
    
    if corrupted_files:
        print(f"\nFound {len(corrupted_files)} corrupted files")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        'organized_dataset', 
        batch_size=8
    )
    
    # Create and train simple neural network
    print("\n=== Training Simple Neural Network ===")
    classifier = CatDogClassifier()
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

if __name__ == "__main__":
    main()