#!/usr/bin/env python3
"""
Inference Script for From Scratch (.pkl) Models
Specifically designed for CatDogClassifierFromScratch models
"""

import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('TrainingScripts')
from DataPreprocess import DataPreprocessor

# Import and define all necessary classes for pickle deserialization
class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

class BatchNormalization:
    """Batch normalization layer implementation"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # For backpropagation
        self.x_normalized = None
        self.var = None
        self.std = None
        
    def forward(self, x, training=True):
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        self.var = var
        self.std = np.sqrt(var + self.eps)
        self.x_normalized = (x - mean) / self.std
        
        # Scale and shift
        return self.gamma * self.x_normalized + self.beta
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        
        # Gradient w.r.t gamma and beta
        grad_gamma = np.sum(grad_output * self.x_normalized, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t input
        grad_x_normalized = grad_output * self.gamma
        grad_var = np.sum(grad_x_normalized * (self.x_normalized) * (-0.5) * (self.var + self.eps)**(-1.5), axis=0)
        grad_mean = np.sum(grad_x_normalized * (-1 / self.std), axis=0) + grad_var * np.sum(-2 * (self.x_normalized * self.std + self.running_mean), axis=0) / batch_size
        
        grad_input = grad_x_normalized / self.std + grad_var * 2 * (self.x_normalized * self.std + self.running_mean) / batch_size + grad_mean / batch_size
        
        return grad_input, grad_gamma, grad_beta

class Dropout:
    """Dropout layer implementation"""
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output):
        return grad_output * self.mask

class DenseLayer:
    """Fully connected layer implementation"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        # For backpropagation
        self.input = None
        
        # For optimization
        self.weights_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output):
        # Gradient w.r.t weights and bias
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t input
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input, grad_weights, grad_bias

class CatDogClassifierFromScratch:
    """Simplified Neural Network implementation for inference"""
    
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)  # 224 * 224 * 1 = 50176
        
        # Initialize empty lists (will be loaded from pickle)
        self.layers = []
        self.dropouts = []
        self.batch_norms = []  # Add batch normalization support
        self.history = {}
        
    def flatten_input(self, x):
        """Flatten input images to 1D"""
        return x.reshape(x.shape[0], -1)
    
    def forward(self, x, training=False):
        """Forward pass through the network (inference mode)"""
        # Flatten input
        x = self.flatten_input(x)
        
        # Layer 1: Dense(256) + BatchNorm + ReLU + Dropout
        x = self.layers[0].forward(x)
        if len(self.batch_norms) > 0 and self.batch_norms[0] is not None:
            x = self.batch_norms[0].forward(x, training)
        x = ActivationFunctions.relu(x)
        if len(self.dropouts) > 0:
            x = self.dropouts[0].forward(x, training)
        
        # Layer 2: Dense(64) + BatchNorm + ReLU + Dropout
        x = self.layers[1].forward(x)
        if len(self.batch_norms) > 1 and self.batch_norms[1] is not None:
            x = self.batch_norms[1].forward(x, training)
        x = ActivationFunctions.relu(x)
        if len(self.dropouts) > 1:
            x = self.dropouts[1].forward(x, training)
        
        # Output layer: Dense(1) + Sigmoid
        x = self.layers[2].forward(x)
        x = ActivationFunctions.sigmoid(x)
        
        return x
    
    def get_total_parameters(self):
        """Calculate and return the total number of trainable parameters"""
        total_params = 0
        for layer in self.layers:
            total_params += layer.weights.size + layer.bias.size
        
        # Add batch norm parameters if present
        for bn in self.batch_norms:
            if bn is not None:
                total_params += bn.gamma.size + bn.beta.size
        
        return total_params

def load_and_evaluate_pkl_model(model_path, organized_dataset_path='organized_dataset'):
    """
    Load a saved .pkl model and evaluate it on test data
    
    Args:
        model_path (str): Path to the .pkl model file
        organized_dataset_path (str): Path to organized dataset directory
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    print("="*60)
    print("FROM SCRATCH MODEL (.PKL) INFERENCE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return None
    
    # Load the saved model
    print(f"Loading model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct model from saved data
        model = CatDogClassifierFromScratch()
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            # Model saved as dictionary
            model.layers = model_data.get('layers', [])
            model.dropouts = model_data.get('dropouts', [])
            model.batch_norms = model_data.get('batch_norms', [])
            model.history = model_data.get('history', {})
            model.input_shape = model_data.get('input_shape', (224, 224, 1))
        else:
            # Model saved as complete object - extract attributes
            if hasattr(model_data, 'layers'):
                model.layers = model_data.layers
            if hasattr(model_data, 'dropouts'):
                model.dropouts = model_data.dropouts
            if hasattr(model_data, 'batch_norms'):
                model.batch_norms = model_data.batch_norms
            else:
                model.batch_norms = []  # No batch norm in this model
            if hasattr(model_data, 'history'):
                model.history = model_data.history
            if hasattr(model_data, 'input_shape'):
                model.input_shape = model_data.input_shape
        
        # Validate model structure
        if not model.layers or len(model.layers) != 3:
            raise ValueError(f"Invalid model structure: expected 3 layers, got {len(model.layers)}")
        
        print("Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Number of layers: {len(model.layers)}")
        print(f"   Total parameters: {model.get_total_parameters():,}")
        print(f"   Architecture: {model.input_shape[0]*model.input_shape[1]*model.input_shape[2]} -> 256 -> 64 -> 1")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None
    
    # Initialize data preprocessor
    print(f"\nPreparing test data...")
    try:
        preprocessor = DataPreprocessor(data_path="Dataset", target_size=(224, 224))
        
        # Create data generators (same settings as training)
        _, _, test_generator = preprocessor.create_data_generators(
            organized_dataset_path,
            batch_size=32
        )
        
        print(f"Test data prepared successfully!")
        print(f"   Test samples: {test_generator.samples}")
        print(f"   Classes: {list(test_generator.class_indices.keys())}")
        
    except Exception as e:
        print(f"Error preparing test data: {e}")
        return None
    
    # Run inference
    print(f"\nRunning inference on {test_generator.samples} test samples...")
    predictions_prob = []
    true_labels = []
    
    try:
        # Reset generator
        try:
            test_generator.reset()
        except AttributeError:
            pass
        
        print(f"Processing {len(test_generator)} batches...")
        
        for batch_idx in range(len(test_generator)):
            if batch_idx % 10 == 0:
                progress = (batch_idx / len(test_generator)) * 100
                print(f"\r  Progress: {progress:.1f}%", end="", flush=True)
            
            x_batch, y_batch = test_generator[batch_idx]
            
            # Run forward pass (inference mode)
            y_pred = model.forward(x_batch, training=False)
            
            predictions_prob.extend(y_pred.flatten())
            true_labels.extend(y_batch.flatten())
        
        print(f"\r  Progress: 100.0%")
        
        predictions_prob = np.array(predictions_prob)
        true_labels = np.array(true_labels)
        
        print(f"\nInference completed!")
        print(f"   Predictions shape: {predictions_prob.shape}")
        print(f"   True labels shape: {true_labels.shape}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    # Calculate metrics
    print(f"\nCalculating performance metrics...")
    
    # Convert probabilities to binary predictions
    predictions_binary = (predictions_prob > 0.5).astype(int)
    
    # Basic counts
    total_samples = len(true_labels)
    correct_predictions = np.sum(predictions_binary == true_labels)
    accuracy = correct_predictions / total_samples
    
    # Confusion matrix components (0 = Cat, 1 = Dog)
    true_negatives = np.sum((predictions_binary == 0) & (true_labels == 0))  # Cats correct
    true_positives = np.sum((predictions_binary == 1) & (true_labels == 1))  # Dogs correct
    false_negatives = np.sum((predictions_binary == 0) & (true_labels == 1)) # Dogs as Cats
    false_positives = np.sum((predictions_binary == 1) & (true_labels == 0)) # Cats as Dogs
    
    # Derived metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Class-wise accuracy
    cat_accuracy = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    dog_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Store results
    results = {
        'model_path': model_path,
        'model_type': 'from_scratch_pkl',
        'total_samples': total_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'cat_accuracy': cat_accuracy,
        'dog_accuracy': dog_accuracy,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'predictions_prob': predictions_prob,
        'predictions_binary': predictions_binary,
        'true_labels': true_labels
    }
    
    return results

def print_performance_report(results):
    """Print a comprehensive performance report"""
    
    print("\n" + "="*70)
    print("FROM SCRATCH MODEL PERFORMANCE REPORT")
    print("="*70)
    
    print(f"\nModel: {results['model_path']}")
    print(f"Model Type: {results['model_type']}")
    print(f"Total Test Samples: {results['total_samples']}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"   - Overall Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   - F1-Score:          {results['f1_score']:.4f}")
    
    print(f"\nDETAILED METRICS:")
    print(f"   - Precision (Dogs):  {results['precision']:.4f}")
    print(f"   - Recall (Dogs):     {results['recall']:.4f}")
    print(f"   - Specificity (Cats):{results['specificity']:.4f}")
    
    print(f"\nCLASS-WISE PERFORMANCE:")
    print(f"   - Cat Accuracy:      {results['cat_accuracy']:.4f} ({results['cat_accuracy']*100:.2f}%)")
    print(f"   - Dog Accuracy:      {results['dog_accuracy']:.4f} ({results['dog_accuracy']*100:.2f}%)")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"   - True Positives:    {results['true_positives']} (Dogs correctly identified)")
    print(f"   - True Negatives:    {results['true_negatives']} (Cats correctly identified)")
    print(f"   - False Positives:   {results['false_positives']} (Cats misclassified as Dogs)")
    print(f"   - False Negatives:   {results['false_negatives']} (Dogs misclassified as Cats)")
    
    # Prediction distribution
    prob_mean = np.mean(results['predictions_prob'])
    prob_std = np.std(results['predictions_prob'])
    prob_min = np.min(results['predictions_prob'])
    prob_max = np.max(results['predictions_prob'])
    
    print(f"\nPREDICTION DISTRIBUTION:")
    print(f"   - Mean probability:  {prob_mean:.4f}")
    print(f"   - Std deviation:     {prob_std:.4f}")
    print(f"   - Min probability:   {prob_min:.4f}")
    print(f"   - Max probability:   {prob_max:.4f}")
    
    print(f"\n" + "="*70)

def main():
    """Main function"""
    
    # Default model path
    model_path = "simple_cat_dog_model_from_scratch_best.pkl"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Run inference
    results = load_and_evaluate_pkl_model(model_path)
    
    if results is not None:
        # Print comprehensive report
        print_performance_report(results)
        
        print(f"\nINFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Final Accuracy: {results['accuracy']:.1%}")
        print(f"   F1-Score: {results['f1_score']:.3f}")
        
    else:
        print(f"\nINFERENCE FAILED!")
        print(f"   Please check the model path and data directory.")

if __name__ == "__main__":
    main()
