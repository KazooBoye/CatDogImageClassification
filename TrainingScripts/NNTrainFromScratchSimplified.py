import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataPreprocess import DataPreprocessor
from PIL import Image

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
        self.x_norm = None
        self.x_centered = None
        self.std = None
        
    def forward(self, x, training=True):
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            self.x_centered = x - batch_mean
            self.std = np.sqrt(batch_var + self.eps)
            self.x_norm = self.x_centered / self.std
            
            # Scale and shift
            output = self.gamma * self.x_norm + self.beta
        else:
            # Use running statistics for inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            output = self.gamma * x_norm + self.beta
            
        return output
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        
        # Gradients w.r.t gamma and beta
        grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t input
        grad_x_norm = grad_output * self.gamma
        grad_variance = np.sum(grad_x_norm * self.x_centered, axis=0) * (-0.5) * (self.std ** -3)
        grad_mean = np.sum(grad_x_norm * (-1 / self.std), axis=0) + grad_variance * np.mean(-2 * self.x_centered, axis=0)
        
        grad_input = (grad_x_norm / self.std) + (grad_variance * 2 * self.x_centered / batch_size) + (grad_mean / batch_size)
        
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
    """Simplified Neural Network implementation optimized for training speed"""
    
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)  # 224 * 224 * 1 = 50176
        
        # Simplified architecture - fewer layers and neurons
        self.layers = []
        self.dropouts = []
        
        # Layer 1: Dense(256) + Dropout(0.3) - Much smaller first layer
        self.layers.append(DenseLayer(self.input_size, 256))
        self.dropouts.append(Dropout(0.3))
        
        # Layer 2: Dense(64) + Dropout(0.2) - Smaller second layer
        self.layers.append(DenseLayer(256, 64))
        self.dropouts.append(Dropout(0.2))
        
        # Output layer: Dense(1) + Sigmoid
        self.layers.append(DenseLayer(64, 1))
        
        # Training history
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
        }
        
        # Optimizer parameters (SGD with momentum - simpler than Adam)
        self.learning_rate = 0.01  # Higher learning rate for faster convergence
        self.momentum = 0.9
        
    def flatten_input(self, x):
        """Flatten input images to 1D"""
        return x.reshape(x.shape[0], -1)
    
    def forward(self, x, training=True):
        """Simplified forward pass through the network"""
        # Flatten input
        x = self.flatten_input(x)
        
        # Layer 1: Dense(256) + ReLU + Dropout
        x = self.layers[0].forward(x)
        x = ActivationFunctions.relu(x)
        x = self.dropouts[0].forward(x, training)
        
        # Layer 2: Dense(64) + ReLU + Dropout
        x = self.layers[1].forward(x)
        x = ActivationFunctions.relu(x)
        x = self.dropouts[1].forward(x, training)
        
        # Output layer: Dense(1) + Sigmoid
        x = self.layers[2].forward(x)
        x = ActivationFunctions.sigmoid(x)
        
        return x
    
    def backward(self, x, y_true, y_pred):
        """Simplified backward pass through the network"""
        batch_size = x.shape[0]
        
        # Calculate loss gradient (binary cross-entropy with sigmoid)
        grad = (y_pred - y_true) / batch_size
        
        # Output layer backward
        grad, grad_weights_2, grad_bias_2 = self.layers[2].backward(grad)
        
        # Layer 2 backward: Dropout + ReLU
        grad = self.dropouts[1].backward(grad)
        # ReLU derivative
        pre_activation_1 = np.dot(self.layers[1].input, self.layers[1].weights) + self.layers[1].bias
        relu_grad_1 = (pre_activation_1 > 0).astype(float)
        grad = grad * relu_grad_1
        grad, grad_weights_1, grad_bias_1 = self.layers[1].backward(grad)
        
        # Layer 1 backward: Dropout + ReLU
        grad = self.dropouts[0].backward(grad)
        # ReLU derivative
        pre_activation_0 = np.dot(self.layers[0].input, self.layers[0].weights) + self.layers[0].bias
        relu_grad_0 = (pre_activation_0 > 0).astype(float)
        grad = grad * relu_grad_0
        grad, grad_weights_0, grad_bias_0 = self.layers[0].backward(grad)
        
        return [
            (grad_weights_0, grad_bias_0),
            (grad_weights_1, grad_bias_1),
            (grad_weights_2, grad_bias_2)
        ]
    
    def update_parameters(self, gradients):
        """Update parameters using SGD with momentum (simpler than Adam)"""
        for i, (grad_weights, grad_bias) in enumerate(gradients):
            layer = self.layers[i]
            
            # Initialize momentum if not exists
            if not hasattr(layer, 'weights_momentum'):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.bias)
            
            # Update momentum
            layer.weights_momentum = self.momentum * layer.weights_momentum - self.learning_rate * grad_weights
            layer.bias_momentum = self.momentum * layer.bias_momentum - self.learning_rate * grad_bias
            
            # Update parameters
            layer.weights += layer.weights_momentum
            layer.bias += layer.bias_momentum
    
    def binary_crossentropy_loss(self, y_true, y_pred):
        """Calculate binary cross-entropy loss"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate accuracy, precision, recall"""
        # Flatten arrays to handle different shapes
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        y_pred_binary = (y_pred_flat > 0.5).astype(int)
        
        # Accuracy
        accuracy = np.mean(y_pred_binary == y_true_flat)
        
        # Precision, Recall, F1
        tp = np.sum((y_pred_binary == 1) & (y_true_flat == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_flat == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_flat == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return accuracy, precision, recall
    
    def train_batch(self, x_batch, y_batch):
        """Train on a single batch"""
        # Ensure y_batch has the correct shape (batch_size, 1)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)
        
        # Forward pass
        y_pred = self.forward(x_batch, training=True)
        
        # Calculate loss
        loss = self.binary_crossentropy_loss(y_batch, y_pred)
        
        # Calculate metrics
        accuracy, precision, recall = self.calculate_metrics(y_batch, y_pred)
        
        # Backward pass
        gradients = self.backward(x_batch, y_batch, y_pred)
        
        # Update parameters
        self.update_parameters(gradients)
        
        return loss, accuracy, precision, recall
    
    def evaluate_batch(self, x_batch, y_batch):
        """Evaluate on a single batch"""
        # Ensure y_batch has the correct shape (batch_size, 1)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)
            
        # Forward pass (no training)
        y_pred = self.forward(x_batch, training=False)
        
        # Calculate loss and metrics
        loss = self.binary_crossentropy_loss(y_batch, y_pred)
        accuracy, precision, recall = self.calculate_metrics(y_batch, y_pred)
        
        return loss, accuracy, precision, recall
    
    def train(self, train_generator, val_generator, epochs=50):  # Reduced epochs
        """Train the simplified model"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Simplified Architecture: {self.input_size} -> 256 -> 64 -> 1")
        
        # Debug: Check first batch to understand data format
        print("\n=== Data Format Check ===")
        try:
            x_sample, y_sample = train_generator[0]
            print(f"Input batch shape: {x_sample.shape}")
            print(f"Output batch shape: {y_sample.shape}")
            print(f"Output batch dtype: {y_sample.dtype}")
            print(f"Output sample values: {y_sample[:5].flatten()}")
        except Exception as e:
            print(f"Error checking data format: {e}")
        
        best_val_accuracy = 0
        patience_counter = 0
        patience = 10  # Reduced patience for faster training
        
        # Track overall training start time
        overall_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses, train_accuracies, train_precisions, train_recalls = [], [], [], []
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            batch_count = 0
            
            # Reset generators (handle potential AttributeError)
            try:
                train_generator.reset()
            except AttributeError:
                # If reset doesn't exist, that's okay
                pass
            
            # Training loop with simplified logging
            training_start_time = time.time()
            print("Training Phase:")
            
            for batch_idx in range(len(train_generator)):
                x_batch, y_batch = train_generator[batch_idx]
                
                # Train on batch
                loss, accuracy, precision, recall = self.train_batch(x_batch, y_batch)
                
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                train_precisions.append(precision)
                train_recalls.append(recall)
                
                batch_count += 1
                progress_percent = (batch_count / len(train_generator)) * 100
                
                # Less frequent logging for speed
                if batch_count % 20 == 0 or batch_count == len(train_generator):
                    # Clear any existing progress line
                    print(f"\r{' ' * 100}\r", end="")
                    
                    avg_loss = np.mean(train_losses[-20:])  # Average of last 20 batches
                    avg_acc = np.mean(train_accuracies[-20:])
                    elapsed_time = time.time() - training_start_time
                    batches_per_sec = batch_count / elapsed_time if elapsed_time > 0 else 0
                    eta = (len(train_generator) - batch_count) / batches_per_sec if batches_per_sec > 0 else 0
                    eta_mins = int(eta / 60)
                    eta_secs = int(eta % 60)
                    
                    print(f"  Batch {batch_count:3d}/{len(train_generator)} ({progress_percent:5.1f}%) - "
                          f"Loss: {loss:.4f} (avg: {avg_loss:.4f}), "
                          f"Acc: {accuracy:.4f} (avg: {avg_acc:.4f}) - "
                          f"Speed: {batches_per_sec:.1f} batch/s, ETA: {eta_mins:02d}:{eta_secs:02d}")
                else:
                    # Minimal progress indication
                    if batch_count % 5 == 0:
                        progress_bars = int(progress_percent / 5)
                        progress_bar = "=" * progress_bars + "-" * (20 - progress_bars)
                        print(f"\r  [{progress_bar}] {progress_percent:5.1f}%", end="", flush=True)
            
            # Clear the progress line and add newline
            print(f"\r{' ' * 100}\r", end="")
            
            training_time = time.time() - training_start_time
            print(f"Training completed in {training_time:.1f}s")
            
            # Simplified validation phase
            print("Validation Phase:")
            val_start_time = time.time()
            val_losses, val_accuracies, val_precisions, val_recalls = [], [], [], []
            
            try:
                val_generator.reset()
            except AttributeError:
                pass
                
            for batch_idx in range(len(val_generator)):
                x_batch, y_batch = val_generator[batch_idx]
                loss, accuracy, precision, recall = self.evaluate_batch(x_batch, y_batch)
                
                val_losses.append(loss)
                val_accuracies.append(accuracy)
                val_precisions.append(precision)
                val_recalls.append(recall)
                
                # Minimal validation progress
                if batch_idx % 10 == 0 or batch_idx == len(val_generator) - 1:
                    val_progress = ((batch_idx + 1) / len(val_generator)) * 100
                    print(f"\r  Validation: {val_progress:5.1f}%", end="", flush=True)
            
            val_time = time.time() - val_start_time
            print(f"\r{' ' * 100}\r", end="")  # Clear progress line
            print(f"Validation completed in {val_time:.1f}s")
            
            # Calculate epoch metrics
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = np.mean(train_accuracies)
            epoch_train_prec = np.mean(train_precisions)
            epoch_train_rec = np.mean(train_recalls)
            
            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = np.mean(val_accuracies)
            epoch_val_prec = np.mean(val_precisions)
            epoch_val_rec = np.mean(val_recalls)
            
            # Calculate epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Store history
            self.history['loss'].append(epoch_train_loss)
            self.history['accuracy'].append(epoch_train_acc)
            self.history['precision'].append(epoch_train_prec)
            self.history['recall'].append(epoch_train_rec)
            
            self.history['val_loss'].append(epoch_val_loss)
            self.history['val_accuracy'].append(epoch_val_acc)
            self.history['val_precision'].append(epoch_val_prec)
            self.history['val_recall'].append(epoch_val_rec)
            
            # Epoch summary with clean formatting
            print(f"\n{'EPOCH SUMMARY':^60}")
            print(f"{'-'*60}")
            print(f"  Training   - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, "
                  f"Prec: {epoch_train_prec:.4f}, Rec: {epoch_train_rec:.4f}")
            print(f"  Validation - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, "
                  f"Prec: {epoch_val_prec:.4f}, Rec: {epoch_val_rec:.4f}")
            print(f"  Time: {epoch_time:.1f}s | Learning Rate: {self.learning_rate:.6f}")
            
            # Loss change indicator
            if len(self.history['val_loss']) > 1:
                loss_change = epoch_val_loss - self.history['val_loss'][-2]
                acc_change = epoch_val_acc - self.history['val_accuracy'][-2]
                loss_trend = "decreasing" if loss_change < 0 else "increasing"
                acc_trend = "increasing" if acc_change > 0 else "decreasing"
                print(f"  Changes: Loss {loss_trend} ({loss_change:+.4f}), "
                      f"Acc {acc_trend} ({acc_change:+.4f})")
            
            print(f"{'-'*60}")
            
            # Early stopping and model saving
            if epoch_val_acc > best_val_accuracy:
                best_val_accuracy = epoch_val_acc
                patience_counter = 0
                self.save_model('simple_cat_dog_model_from_scratch_best.pkl')
                print(f"  >>> NEW BEST! Validation accuracy: {best_val_accuracy:.4f} - Model saved")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epoch(s). Best: {best_val_accuracy:.4f}")
                
            # Learning rate reduction on plateau
            if patience_counter >= 3:  # Reduce LR more aggressively
                old_lr = self.learning_rate
                self.learning_rate *= 0.5
                print(f"  >>> Learning rate reduced: {old_lr:.6f} -> {self.learning_rate:.6f}")
                patience_counter = 0
                
            if patience_counter >= patience:
                print(f"  >>> Early stopping triggered after {patience} epochs without improvement")
                print(f"  >>> Best validation accuracy achieved: {best_val_accuracy:.4f}")
                break
                
            # Show remaining time estimate
            if epoch > 0:
                avg_epoch_time = (time.time() - overall_start_time) / (epoch + 1)
                remaining_epochs = epochs - epoch - 1
                eta_total = avg_epoch_time * remaining_epochs
                eta_hours = int(eta_total // 3600)
                eta_mins = int((eta_total % 3600) // 60)
                if eta_hours > 0:
                    print(f"  Estimated time remaining: {eta_hours}h {eta_mins}m")
                else:
                    print(f"  Estimated time remaining: {eta_mins}m")
            
            print(f"{'='*60}")
    
    def save_model(self, filename):
        """Save simplified model parameters"""
        model_data = {
            'layers': self.layers,
            'dropouts': self.dropouts,
            'history': self.history,
            'input_shape': self.input_shape
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename):
        """Load simplified model parameters"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.layers = model_data['layers']
        self.dropouts = model_data['dropouts']
        self.history = model_data['history']
        self.input_shape = model_data['input_shape']
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history['loss']:
            print("No training history found. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Accuracy
        axes[0, 0].plot(epochs, self.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(epochs, self.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(epochs, self.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(epochs, self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(epochs, self.history['precision'], label='Training Precision', color='blue')
        axes[1, 0].plot(epochs, self.history['val_precision'], label='Validation Precision', color='red')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(epochs, self.history['recall'], label='Training Recall', color='blue')
        axes[1, 1].plot(epochs, self.history['val_recall'], label='Validation Recall', color='red')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('TrainingLogs/training_history_from_scratch.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_generator):
        """Evaluate model on test data"""
        print("Evaluating model on test data...")
        
        test_losses, test_accuracies, test_precisions, test_recalls = [], [], [], []
        
        try:
            test_generator.reset()
        except AttributeError:
            pass
            
        for batch_idx in range(len(test_generator)):
            x_batch, y_batch = test_generator[batch_idx]
            loss, accuracy, precision, recall = self.evaluate_batch(x_batch, y_batch)
            
            test_losses.append(loss)
            test_accuracies.append(accuracy)
            test_precisions.append(precision)
            test_recalls.append(recall)
        
        # Calculate overall metrics
        test_loss = np.mean(test_losses)
        test_accuracy = np.mean(test_accuracies)
        test_precision = np.mean(test_precisions)
        test_recall = np.mean(test_recalls)
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        
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
    
    def get_total_parameters(self):
        """Calculate and return the total number of trainable parameters"""
        total_params = 0
        for layer in self.layers:
            total_params += layer.weights.size + layer.bias.size
        return total_params

def main():
    print("=== Neural Network from Scratch - Cat vs Dog Classifier ===")
    print("Architecture matching NNTrain.py:")
    print("Input(224x224x1) -> Dense(2048) -> Dense(512) -> Dense(128) -> Dense(32) -> Dense(1)")
    print("With ReLU activations, Batch Normalization, and Dropout")
    
    # Initialize preprocessor and create data generators
    preprocessor = DataPreprocessor(data_path="Dataset", target_size=(224, 224))
    
    # Create data generators
    print("\n=== Creating Data Generators ===")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        'organized_dataset', 
        batch_size=32  # Smaller batch size for from-scratch implementation
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # Create and train neural network from scratch
    print("\n=== Training Simplified Neural Network from Scratch ===")
    classifier = CatDogClassifierFromScratch()
    
    print(f"Total parameters: {classifier.get_total_parameters():,}")
    print("Network optimized for training speed over accuracy")
    
    # Train model
    classifier.train(train_gen, val_gen, epochs=100)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    results = classifier.evaluate_model(test_gen)
    
    print(f"\nFinal Results Summary:")
    print(f"Best Test Accuracy: {results['accuracy']:.4f}")
    print(f"Model saved as: simple_cat_dog_model_from_scratch_best.pkl")
    print("Training completed using pure NumPy implementation!")

if __name__ == "__main__":
    main()
