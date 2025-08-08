#!/usr/bin/env python3
"""
Simple Model Inference Script for Cat-Dog Classification
Runs inference on saved .keras model and reports performance metrics
"""

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DataPreprocess import DataPreprocessor

def load_and_evaluate_model(model_path, organized_dataset_path='organized_dataset'):
    """
    Load a saved model and evaluate it on test data
    
    Args:
        model_path (str): Path to the .keras model file
        organized_dataset_path (str): Path to organized dataset directory
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    print("="*60)
    print("CAT-DOG CLASSIFICATION MODEL INFERENCE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("\nAvailable .keras files in current directory:")
        keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        if keras_files:
            for file in keras_files:
                print(f"   - {file}")
        else:
            print("   No .keras files found")
        return None
    
    # Load the saved model
    print(f"Loading model from: {model_path}")
    try:
        model = models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Initialize data preprocessor
    print(f"\nPreparing test data...")
    try:
        preprocessor = DataPreprocessor(data_path="Dataset", target_size=(224, 224))
        
        # Create data generators (same settings as training)
        _, _, test_generator = preprocessor.create_data_generators(
            organized_dataset_path,
            batch_size=32  # Use reasonable batch size for inference
        )
        
        print(f"Test data prepared successfully!")
        print(f"   Test samples: {test_generator.samples}")
        print(f"   Classes: {list(test_generator.class_indices.keys())}")
        print(f"   Class mapping: {test_generator.class_indices}")
        
    except Exception as e:
        print(f"Error preparing test data: {e}")
        return None
    
    # Run inference
    print(f"\nRunning inference on {test_generator.samples} test samples...")
    try:
        # Reset generator to ensure we start from beginning
        test_generator.reset()
        
        # Get predictions
        print("   Making predictions...")
        predictions_prob = model.predict(
            test_generator,
            steps=len(test_generator),
            verbose=1
        )
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
        
        # Get true labels
        true_labels = test_generator.classes
        
        print(f"Inference completed!")
        print(f"   Prediction probabilities shape: {predictions_prob.shape}")
        print(f"   Binary predictions shape: {predictions_binary.shape}")
        print(f"   True labels shape: {true_labels.shape}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    # Calculate detailed metrics
    print(f"\nCalculating performance metrics...")
    
    # Basic counts
    total_samples = len(true_labels)
    correct_predictions = np.sum(predictions_binary == true_labels)
    accuracy = correct_predictions / total_samples
    
    # Confusion matrix components
    # Note: 0 = Cat, 1 = Dog
    true_negatives = np.sum((predictions_binary == 0) & (true_labels == 0))  # Correctly predicted Cats
    true_positives = np.sum((predictions_binary == 1) & (true_labels == 1))  # Correctly predicted Dogs
    false_negatives = np.sum((predictions_binary == 0) & (true_labels == 1)) # Dogs predicted as Cats
    false_positives = np.sum((predictions_binary == 1) & (true_labels == 0)) # Cats predicted as Dogs
    
    # Derived metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Class-wise accuracy
    cat_accuracy = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    dog_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Error analysis
    total_errors = false_positives + false_negatives
    error_rate = total_errors / total_samples
    
    # Store results
    results = {
        'model_path': model_path,
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
        'total_errors': total_errors,
        'error_rate': error_rate,
        'predictions_prob': predictions_prob,
        'predictions_binary': predictions_binary,
        'true_labels': true_labels
    }
    
    return results

def print_performance_report(results):
    """Print a comprehensive performance report"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*70)
    
    print(f"\nModel: {results['model_path']}")
    print(f"Total Test Samples: {results['total_samples']}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"   - Overall Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   - F1-Score:          {results['f1_score']:.4f}")
    print(f"   - Error Rate:        {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)")
    
    print(f"\nDETAILED METRICS:")
    print(f"   - Precision (Dogs):  {results['precision']:.4f}")
    print(f"   - Recall (Dogs):     {results['recall']:.4f}")
    print(f"   - Specificity (Cats):{results['specificity']:.4f}")
    
    print(f"\nCLASS-WISE PERFORMANCE:")
    print(f"   - Cat Accuracy:      {results['cat_accuracy']:.4f} ({results['cat_accuracy']*100:.2f}%)")
    print(f"   - Dog Accuracy:      {results['dog_accuracy']:.4f} ({results['dog_accuracy']*100:.2f}%)")
    
    print(f"\nCONFUSION MATRIX BREAKDOWN:")
    print(f"   - True Positives:    {results['true_positives']} (Dogs correctly identified)")
    print(f"   - True Negatives:    {results['true_negatives']} (Cats correctly identified)")
    print(f"   - False Positives:   {results['false_positives']} (Cats misclassified as Dogs)")
    print(f"   - False Negatives:   {results['false_negatives']} (Dogs misclassified as Cats)")
    
    print(f"\nERROR ANALYSIS:")
    print(f"   - Total Errors:      {results['total_errors']} out of {results['total_samples']}")
    print(f"   - Cat to Dog Errors: {results['false_positives']}")
    print(f"   - Dog to Cat Errors: {results['false_negatives']}")
    
    # Prediction distribution
    prob_mean = np.mean(results['predictions_prob'])
    prob_std = np.std(results['predictions_prob'])
    prob_min = np.min(results['predictions_prob'])
    prob_max = np.max(results['predictions_prob'])
    
    print(f"\nPREDICTION PROBABILITY DISTRIBUTION:")
    print(f"   - Mean:              {prob_mean:.4f}")
    print(f"   - Standard Deviation:{prob_std:.4f}")
    print(f"   - Min:               {prob_min:.4f}")
    print(f"   - Max:               {prob_max:.4f}")
    
    # Confidence analysis
    high_confidence = np.sum((results['predictions_prob'] > 0.8) | (results['predictions_prob'] < 0.2))
    medium_confidence = np.sum((results['predictions_prob'] >= 0.6) & (results['predictions_prob'] <= 0.8)) + \
                       np.sum((results['predictions_prob'] >= 0.2) & (results['predictions_prob'] <= 0.4))
    low_confidence = np.sum((results['predictions_prob'] >= 0.4) & (results['predictions_prob'] <= 0.6))
    
    print(f"\nPREDICTION CONFIDENCE ANALYSIS:")
    print(f"   - High Confidence (>0.8 or <0.2): {high_confidence} ({high_confidence/results['total_samples']*100:.1f}%)")
    print(f"   - Medium Confidence:               {medium_confidence} ({medium_confidence/results['total_samples']*100:.1f}%)")
    print(f"   - Low Confidence (0.4-0.6):       {low_confidence} ({low_confidence/results['total_samples']*100:.1f}%)")
    
    print(f"\n" + "="*70)

def save_results_summary(results, output_file="inference_summary.txt"):
    """Save results summary to a text file"""
    
    with open(output_file, 'w') as f:
        f.write("CAT-DOG CLASSIFICATION MODEL INFERENCE RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Total Test Samples: {results['total_samples']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"Specificity: {results['specificity']:.4f}\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n")
        f.write(f"Error Rate: {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)\n\n")
        
        f.write("CLASS-WISE ACCURACY:\n")
        f.write(f"Cat Accuracy: {results['cat_accuracy']:.4f} ({results['cat_accuracy']*100:.2f}%)\n")
        f.write(f"Dog Accuracy: {results['dog_accuracy']:.4f} ({results['dog_accuracy']*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives (Dogs correct): {results['true_positives']}\n")
        f.write(f"True Negatives (Cats correct): {results['true_negatives']}\n")
        f.write(f"False Positives (Cats→Dogs): {results['false_positives']}\n")
        f.write(f"False Negatives (Dogs→Cats): {results['false_negatives']}\n")
    
    print(f"Results summary saved to: {output_file}")

def main():
    """Main function"""
    
    # Default model path
    model_path = "simple_cat_dog_model_from_scratch_best.pkl"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Run inference
    results = load_and_evaluate_model(model_path)
    
    if results is not None:
        # Print comprehensive report
        print_performance_report(results)
        
        # Save summary
        save_results_summary(results)
        
        print(f"\nINFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Final Accuracy: {results['accuracy']:.1%}")
        print(f"   F1-Score: {results['f1_score']:.3f}")
        
    else:
        print(f"\nINFERENCE FAILED!")
        print(f"   Please check the model path and data directory.")

if __name__ == "__main__":
    main()
