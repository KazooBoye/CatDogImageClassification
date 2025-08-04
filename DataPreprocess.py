import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

class DataPreprocessor:
    def __init__(self, data_path, target_size=(224, 224)):
        self.data_path = data_path
        self.target_size = target_size
        
    def validate_images(self, folder_path):
        valid_images = []
        corrupted_count = 0
        
        for filename in os.listdir(folder_path):
            # Skip macOS metadata files and hidden files
            if filename.startswith('.') or filename.startswith('._'):
                continue
                
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                try:
                    # Try to open and load image completely
                    with Image.open(filepath) as img:
                        # Verify the image can be loaded
                        img.load()
                        
                        # Check if image has reasonable dimensions
                        if img.size[0] > 50 and img.size[1] > 50:
                            # Try to convert to RGB to ensure it's processable
                            img.convert('RGB')
                            valid_images.append(filepath)
                        else:
                            print(f"Small image skipped: {filename} (size: {img.size})")
                            corrupted_count += 1
                            
                except Exception as e:
                    print(f"Corrupted image skipped: {filename} - {str(e)}")
                    corrupted_count += 1
                        
        print(f"Found {len(valid_images)} valid images, {corrupted_count} problematic files")
        return valid_images

    def preprocess_and_save_image(self, input_path, output_path):
        """Preprocess a single image and save it"""
        try:
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Could not load image")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size)
            
            # Normalize pixel values to [0, 1] and convert back to [0, 255] for saving
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_to_save = (img_normalized * 255).astype(np.uint8)
            
            # Convert back to PIL for saving
            pil_img = Image.fromarray(img_to_save)
            pil_img.save(output_path, 'JPEG', quality=95)
            
            return True
            
        except Exception as e:
            print(f"Error preprocessing {input_path}: {e}")
            return False

    def _process_image_list(self, image_list, output_folder):
        """Helper method to preprocess and save a list of images"""
        successful = 0
        for img_path in image_list:
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, filename)
            if self.preprocess_and_save_image(img_path, output_path):
                successful += 1
        print(f"  Successfully processed: {successful}/{len(image_list)} images")

    def organize_and_preprocess_dataset(self, cats_folder, dogs_folder, output_dir, test_split=0.1, val_split=0.2):
        """Organize dataset into train/val/test structure with preprocessing"""
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{output_dir}/{split}/cats", exist_ok=True)
            os.makedirs(f"{output_dir}/{split}/dogs", exist_ok=True)
        
        # Process cats
        print("Processing cat images...")
        cat_images = self.validate_images(cats_folder)
        
        # Split: first test, then train/val from remaining
        cats_temp, cats_test = train_test_split(cat_images, test_size=test_split, random_state=42)
        cats_train, cats_val = train_test_split(cats_temp, test_size=val_split/(1-test_split), random_state=42)
        
        # Process dogs
        print("Processing dog images...")
        dog_images = self.validate_images(dogs_folder)
        
        # Split: first test, then train/val from remaining
        dogs_temp, dogs_test = train_test_split(dog_images, test_size=test_split, random_state=42)
        dogs_train, dogs_val = train_test_split(dogs_temp, test_size=val_split/(1-test_split), random_state=42)
        
        # Preprocess and copy files to organized structure
        print("Preprocessing and copying cat images...")
        print("  Training set:")
        self._process_image_list(cats_train, f"{output_dir}/train/cats/")
        print("  Validation set:")
        self._process_image_list(cats_val, f"{output_dir}/val/cats/")
        print("  Test set:")
        self._process_image_list(cats_test, f"{output_dir}/test/cats/")
        
        print("Preprocessing and copying dog images...")
        print("  Training set:")
        self._process_image_list(dogs_train, f"{output_dir}/train/dogs/")
        print("  Validation set:")
        self._process_image_list(dogs_val, f"{output_dir}/val/dogs/")
        print("  Test set:")
        self._process_image_list(dogs_test, f"{output_dir}/test/dogs/")
        
        print(f"\nDataset organized and preprocessed:")
        print(f"Train: {len(cats_train)} cats, {len(dogs_train)} dogs")
        print(f"Val:   {len(cats_val)} cats, {len(dogs_val)} dogs")
        print(f"Test:  {len(cats_test)} cats, {len(dogs_test)} dogs")

    def create_data_generators(self, data_dir, batch_size=32):
        """Create data generators with augmentation for preprocessed images"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Still need this since we saved as [0,255]
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            f"{data_dir}/train",
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Validation generator
        val_generator = val_test_datagen.flow_from_directory(
            f"{data_dir}/val",
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Test generator
        test_generator = val_test_datagen.flow_from_directory(
            f"{data_dir}/test",
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False  # Don't shuffle test data
        )
        
        return train_generator, val_generator, test_generator



# Usage example
if __name__ == "__main__":
    preprocessor = DataPreprocessor(data_path="./Dataset", target_size=(224, 224))
    
    # Organize and preprocess dataset (10% test, 20% val, 70% train)
    preprocessor.organize_and_preprocess_dataset(
        './Dataset/Cat/', 
        './Dataset/Dog/', 
        'organized_dataset',
        test_split=0.1,
        val_split=0.2
    )
    
    # Create data generators
    # train_gen, val_gen, test_gen = preprocessor.create_data_generators('organized_dataset')
