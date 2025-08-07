import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from pathlib import Path

class ActivationFunctions:
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
    def __init__(self, num_features, epsilon=1e-8, momentum=0.99):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))  # Scale
        self.beta = np.zeros((1, num_features))  # Shift
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backprop
        self.cache = {}
        self.training = True
    
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Cache for backprop
            self.cache = {
                'x': x,
                'x_norm': x_norm,
                'batch_mean': batch_mean,
                'batch_var': batch_var
            }
        else:
            # Use running statistics for inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out
    
    def backward(self, dout):
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        
        N = x.shape[0]
        
        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for x
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - batch_mean) * -0.5 * (batch_var + self.epsilon)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (x - batch_mean), axis=0, keepdims=True) / N
        dx = dx_norm / np.sqrt(batch_var + self.epsilon) + dvar * 2 * (x - batch_mean) / N + dmean / N
        
        return dx

class Dropout:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=x.shape) / (1 - self.drop_rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # Cache for backpropagation
        self.cache = {}
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x):
        # Linear transformation
        z = np.dot(x, self.weights) + self.biases
        
        # Apply activation
        if self.activation == 'relu':
            a = ActivationFunctions.relu(z)
        elif self.activation == 'sigmoid':
            a = ActivationFunctions.sigmoid(z)
        else:  # linear
            a = z
        
        # Cache for backprop
        self.cache = {'x': x, 'z': z, 'a': a}
        return a
    
    def backward(self, dout):
        x = self.cache['x']
        z = self.cache['z']
        
        # Activation derivative
        if self.activation == 'relu':
            dz = dout * ActivationFunctions.relu_derivative(z)
        elif self.activation == 'sigmoid':
            dz = dout * ActivationFunctions.sigmoid_derivative(z)
        else:  # linear
            dz = dout
        
        # Gradients
        self.dW = np.dot(x.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        # Gradient w.r.t input
        dx = np.dot(dz, self.weights.T)
        return dx

# Memory-efficient data generator (no GPU support - pure NumPy)
class DataGenerator:
    """Memory-efficient data loader that loads batches on-demand"""
    def __init__(self, data_dir, batch_size=32, target_size=(224, 224), class_mode='binary'):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load cat images (label 0)
        cat_dir = self.data_dir / 'cats'
        if cat_dir.exists():
            for img_path in cat_dir.glob('*.jpg'):
                self.image_paths.append(str(img_path))
                self.labels.append(0)
        
        # Load dog images (label 1)
        dog_dir = self.data_dir / 'dogs'
        if dog_dir.exists():
            for img_path in dog_dir.glob('*.jpg'):
                self.image_paths.append(str(img_path))
                self.labels.append(1)
        
        self.labels = np.array(self.labels).reshape(-1, 1)
        self.n_samples = len(self.image_paths)
        
        print(f"Found {self.n_samples} images in {data_dir}")
        print(f"  Cats: {np.sum(self.labels == 0)}")
        print(f"  Dogs: {np.sum(self.labels == 1)}")
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img) / 255.0  # Normalize to [0,1]
            return img_array
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zeros as fallback
            return np.zeros((*self.target_size, 3))
    
    def get_batch(self, batch_idx):
        """Get a specific batch of data"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # Load images for this batch
        batch_images = []
        for img_path in batch_paths:
            img = self.load_and_preprocess_image(img_path)
            batch_images.append(img)
        
        return np.array(batch_images), batch_labels
    
    def flow(self, shuffle=True):
        """Generator that yields batches indefinitely"""
        while True:
            if shuffle:
                # Shuffle indices
                indices = np.random.permutation(self.n_samples)
                shuffled_paths = [self.image_paths[i] for i in indices]
                shuffled_labels = self.labels[indices]
            else:
                shuffled_paths = self.image_paths
                shuffled_labels = self.labels
            
            # Yield batches
            for i in range(0, self.n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, self.n_samples)
                batch_paths = shuffled_paths[i:end_idx]
                batch_labels = shuffled_labels[i:end_idx]
                
                # Load images for this batch
                batch_images = []
                for img_path in batch_paths:
                    img = self.load_and_preprocess_image(img_path)
                    batch_images.append(img)
                
                yield np.array(batch_images), batch_labels

class NeuralNetworkFromScratch:
    def __init__(self, input_size=150528):  # 224*224*3
        self.input_size = input_size
        self.layers = []
        self.batch_norms = []
        self.dropouts = []
        self.build_model()
        
        # Training parameters
        self.learning_rate = 0.001
        self.beta1 = 0.9  # Adam parameter
        self.beta2 = 0.999  # Adam parameter
        self.epsilon = 1e-8  # Adam parameter
        self.t = 0  # Time step for Adam
        
        # Adam optimizer variables
        self.m_weights = []  # First moment
        self.v_weights = []  # Second moment
        self.m_biases = []
        self.v_biases = []
        self.m_gamma = []  # For batch norm gamma
        self.v_gamma = []
        self.m_beta = []   # For batch norm beta
        self.v_beta = []
        
        self.initialize_adam()
    
    def build_model(self):
        """Build the same architecture as TensorFlow model"""
        # Layer 1: 512 neurons + ReLU + BatchNorm + Dropout(0.5)
        self.layers.append(DenseLayer(self.input_size, 512, 'relu'))
        self.batch_norms.append(BatchNormalization(512))
        self.dropouts.append(Dropout(0.5))
        
        # Layer 2: 256 neurons + ReLU + BatchNorm + Dropout(0.4)
        self.layers.append(DenseLayer(512, 256, 'relu'))
        self.batch_norms.append(BatchNormalization(256))
        self.dropouts.append(Dropout(0.4))
        
        # Layer 3: 128 neurons + ReLU + BatchNorm + Dropout(0.3)
        self.layers.append(DenseLayer(256, 128, 'relu'))
        self.batch_norms.append(BatchNormalization(128))
        self.dropouts.append(Dropout(0.3))
        
        # Layer 4: 64 neurons + ReLU + Dropout(0.2)
        self.layers.append(DenseLayer(128, 64, 'relu'))
        self.dropouts.append(Dropout(0.2))
        
        # Output layer: 1 neuron + Sigmoid
        self.layers.append(DenseLayer(64, 1, 'sigmoid'))
    
    def initialize_adam(self):
        """Initialize Adam optimizer variables"""
        for layer in self.layers:
            self.m_weights.append(np.zeros_like(layer.weights))
            self.v_weights.append(np.zeros_like(layer.weights))
            self.m_biases.append(np.zeros_like(layer.biases))
            self.v_biases.append(np.zeros_like(layer.biases))
        
        for bn in self.batch_norms:
            self.m_gamma.append(np.zeros_like(bn.gamma))
            self.v_gamma.append(np.zeros_like(bn.gamma))
            self.m_beta.append(np.zeros_like(bn.beta))
            self.v_beta.append(np.zeros_like(bn.beta))
    
    def forward(self, x):
        """Forward propagation"""
        current_input = x.reshape(x.shape[0], -1)  # Flatten
        
        # First 3 layers with batch norm
        for i in range(3):
            current_input = self.layers[i].forward(current_input)
            current_input = self.batch_norms[i].forward(current_input)
            current_input = self.dropouts[i].forward(current_input)
        
        # Layer 4 (no batch norm)
        current_input = self.layers[3].forward(current_input)
        current_input = self.dropouts[3].forward(current_input)
        
        # Output layer
        output = self.layers[4].forward(current_input)
        
        return output
    
    def backward(self, dout):
        """Backward propagation"""
        # Output layer
        dout = self.layers[4].backward(dout)
        
        # Layer 4 (no batch norm)
        dout = self.dropouts[3].backward(dout)
        dout = self.layers[3].backward(dout)
        
        # First 3 layers with batch norm (reverse order)
        for i in range(2, -1, -1):
            dout = self.dropouts[i].backward(dout)
            dout = self.batch_norms[i].backward(dout)
            dout = self.layers[i].backward(dout)
    
    def binary_cross_entropy_loss(self, y_pred, y_true):
        """Binary cross entropy loss"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def binary_cross_entropy_derivative(self, y_pred, y_true):
        """Derivative of binary cross entropy loss"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
    
    def set_training_mode(self, training=True):
        """Set training/inference mode"""
        for bn in self.batch_norms:
            bn.training = training
        for dropout in self.dropouts:
            dropout.training = training
    
    def update_parameters_adam(self):
        """Update parameters using Adam optimizer"""
        self.t += 1
        
        # Update layer weights and biases
        for i, layer in enumerate(self.layers):
            # Weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * layer.dW
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (layer.dW ** 2)
            
            m_hat = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_weights[i] / (1 - self.beta2 ** self.t)
            
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * layer.db
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (layer.db ** 2)
            
            m_hat = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update batch norm parameters
        for i, bn in enumerate(self.batch_norms):
            # Gamma
            self.m_gamma[i] = self.beta1 * self.m_gamma[i] + (1 - self.beta1) * bn.dgamma
            self.v_gamma[i] = self.beta2 * self.v_gamma[i] + (1 - self.beta2) * (bn.dgamma ** 2)
            
            m_hat = self.m_gamma[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_gamma[i] / (1 - self.beta2 ** self.t)
            
            bn.gamma -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Beta
            self.m_beta[i] = self.beta1 * self.m_beta[i] + (1 - self.beta1) * bn.dbeta
            self.v_beta[i] = self.beta2 * self.v_beta[i] + (1 - self.beta2) * (bn.dbeta ** 2)
            
            m_hat = self.m_beta[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_beta[i] / (1 - self.beta2 ** self.t)
            
            bn.beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def train_batch(self, x_batch, y_batch):
        """Train on a single batch"""
        # Forward pass
        y_pred = self.forward(x_batch)
        
        # Calculate loss
        loss = self.binary_cross_entropy_loss(y_pred, y_batch)
        
        # Backward pass
        dout = self.binary_cross_entropy_derivative(y_pred, y_batch)
        self.backward(dout)
        
        # Update parameters
        self.update_parameters_adam()
        
        # Calculate accuracy
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y_batch)
        
        return loss, accuracy
    
    def predict(self, x):
        """Make predictions"""
        self.set_training_mode(False)
        y_pred = self.forward(x)
        self.set_training_mode(True)
        return y_pred
    
    def evaluate_generator(self, generator, steps):
        """Evaluate model using a data generator"""
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        for step in range(steps):
            x_batch, y_batch = generator.get_batch(step)
            y_pred = self.predict(x_batch)
            
            loss = self.binary_cross_entropy_loss(y_pred, y_batch)
            predictions = (y_pred > 0.5).astype(int)
            accuracy = np.mean(predictions == y_batch)
            
            batch_size = x_batch.shape[0]
            total_loss += loss * batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size
            
            all_predictions.extend(predictions.flatten())
            all_labels.extend(y_batch.flatten())
        
        # Calculate overall metrics
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        # Calculate precision, recall, F1
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def count_parameters(self):
        """Count total parameters"""
        total = 0
        for layer in self.layers:
            total += layer.weights.size + layer.biases.size
        for bn in self.batch_norms:
            total += bn.gamma.size + bn.beta.size
        return total
    
    def save_model(self, filepath):
        """Save model parameters"""
        model_data = {
            'layers': [],
            'batch_norms': [],
            'input_size': self.input_size
        }
        
        for layer in self.layers:
            model_data['layers'].append({
                'weights': layer.weights,
                'biases': layer.biases,
                'activation': layer.activation
            })
        
        for bn in self.batch_norms:
            model_data['batch_norms'].append({
                'gamma': bn.gamma,
                'beta': bn.beta,
                'running_mean': bn.running_mean,
                'running_var': bn.running_var
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

def train_model_from_scratch():
    """Main training function with memory-efficient data loading"""
    # Create memory-efficient data generators
    data_dir = 'organized_dataset'
    batch_size = 16  # Small batch size for memory efficiency
    
    train_gen = DataGenerator(f'{data_dir}/train', batch_size=batch_size)
    val_gen = DataGenerator(f'{data_dir}/val', batch_size=batch_size)
    test_gen = DataGenerator(f'{data_dir}/test', batch_size=batch_size)
    
    # Calculate steps per epoch
    train_steps = len(train_gen)
    val_steps = len(val_gen)
    test_steps = len(test_gen)
    
    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    print(f"Test steps: {test_steps}")
    
    # Create model
    model = NeuralNetworkFromScratch()
    print(f"Model created with {model.count_parameters():,} parameters")
    print("⚠️  NOTE: This implementation uses CPU only (no GPU acceleration)")
    
    # Training parameters
    epochs = 20  # Reduced for CPU training
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print("Memory-efficient loading: Only one batch in memory at a time")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        epoch_losses = []
        epoch_accuracies = []
        
        # Create a fresh generator for this epoch
        train_gen_flow = train_gen.flow(shuffle=True)
        
        for step in range(train_steps):
            x_batch, y_batch = next(train_gen_flow)
            
            loss, accuracy = model.train_batch(x_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            
            # Print progress every 50 steps
            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{train_steps} - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        # Validation (memory-efficient)
        print("  Validating...")
        val_metrics = model.evaluate_generator(val_gen, val_steps)
        
        # Store history
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        # Print epoch results
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    
    # Note: For training evaluation, we use a subset to save memory
    print("Evaluating on training set (subset)...")
    train_subset_steps = min(100, train_steps)  # Evaluate on first 100 batches
    train_metrics = model.evaluate_generator(train_gen, train_subset_steps)
    
    print("Evaluating on validation set...")
    val_metrics = model.evaluate_generator(val_gen, val_steps)
    
    print("Evaluating on test set...")
    test_metrics = model.evaluate_generator(test_gen, test_steps)
    
    print(f"Train (subset) - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val            - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    print(f"Test           - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test           - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1_score']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('scratch_model_training_history_efficient.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model.save_model('scratch_cat_dog_model_efficient.pkl')
    
    return model, test_metrics

if __name__ == "__main__":
    print("=== Neural Network From Scratch (Memory-Efficient) ===")
    print("This implementation:")
    print("✅ Uses memory-efficient batch loading")
    print("❌ NO GPU acceleration (pure NumPy/CPU)")
    print("⚠️  Will be much slower than TensorFlow GPU version")
    print("\nRecommendation: Use your TensorFlow CNNTrainImproved.py for best results!")
    
    # Ask user if they want to continue
    response = input("\nDo you want to continue with CPU-only training? (y/n): ")
    if response.lower() == 'y':
        # Train the model
        model, results = train_model_from_scratch()
        print(f"\nTraining completed!")
        print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    else:
        print("Training cancelled. Use CNNTrainImproved.py for GPU-accelerated training!")
