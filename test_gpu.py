#!/usr/bin/env python3
"""
Test script to verify GPU setup for neural network training
"""

import tensorflow as tf
import os

def test_gpu_setup():
    """Test GPU configuration and availability"""
    print("=== TensorFlow GPU Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth configured")
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
        
        # Test GPU computation
        print("\n=== Testing GPU Computation ===")
        try:
            with tf.device('/GPU:0'):
                # Create some test tensors
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                
                # Perform computation
                c = tf.matmul(a, b)
                result = tf.reduce_sum(c)
                
            print(f"✓ GPU computation successful: {result.numpy():.2f}")
            
            # Test mixed precision
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✓ Mixed precision policy set successfully")
                
                # Reset to default for compatibility
                tf.keras.mixed_precision.set_global_policy('float32')
                
            except Exception as e:
                print(f"✗ Mixed precision test failed: {e}")
                
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
    else:
        print("✗ No GPU available - will use CPU")
    
    # Test simple model creation on GPU
    print("\n=== Testing Model Creation ===")
    try:
        device = '/GPU:0' if gpus else '/CPU:0'
        print(f"Creating model on: {device}")
        
        with tf.device(device):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        print(f"✓ Model created successfully with {model.count_params()} parameters")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
    
    print("\n=== Summary ===")
    if gpus:
        print("✓ GPU is available and configured for training")
        print("Your neural network training will utilize GPU acceleration")
    else:
        print("✗ No GPU available - training will use CPU only")

if __name__ == "__main__":
    test_gpu_setup()
