import numpy as np


class FNN:
    """
    Feedforward Neural Network (Fully Connected Network).
    
    A sequential model where layers are stacked and data flows
    from input to output through each layer in order.
    """
    
    def __init__(self, loss, optimizer):
        """
        Initialize the FNN model.
        
        Args:
            loss: Loss function instance with forward and backward methods.
            optimizer: Optimizer instance with step method.
        """
        self.layers = []
        self.optimizer = optimizer
        self.loss = loss
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def add_layer(self, layer):
        """
        Add a layer to the model.
        
        Args:
            layer: A layer object with forward and backward methods.
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Perform a forward pass through the model.
        
        Args:
            X: Input data of shape (batch_size, input_features).
        
        Returns:
            Output of the model after passing through all layers.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        """
        Perform a backward pass through the model.
        
        Args:
            dA: Gradient of the loss with respect to the model output.
        """
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def _set_training_mode(self, training=True):
        """
        Set all layers to training or evaluation mode.
        
        Args:
            training: If True, set to training mode; otherwise evaluation mode.
        """
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training

    def train(self, X, y, epochs, batch_size, validation_data=None, validation_split=0.0, verbose=True):
        """
        Train the model using mini-batch gradient descent.
        
        Args:
            X: Training data of shape (num_samples, input_features).
            y: Training labels of shape (num_samples, num_classes).
            epochs: Number of training epochs.
            batch_size: Size of each mini-batch.
            validation_data: Optional tuple of (X_val, y_val) for validation.
            validation_split: Fraction of training data to use for validation (0.0 to 1.0).
            verbose: Whether to print training progress.
        
        Returns:
            Dictionary containing training history.
        """
        if validation_split > 0 and validation_data is None:
            split_idx = int(X.shape[0] * (1 - validation_split))
            indices = np.random.permutation(X.shape[0])
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            X_val, y_val = X[val_indices], y[val_indices]
            X, y = X[train_indices], y[train_indices]
            validation_data = (X_val, y_val)
        
        num_samples = X.shape[0]
        
        for epoch in range(epochs):
            self._set_training_mode(True)
            
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                y_pred = self.forward(X_batch)
                
                loss = self.loss.forward(y_batch, y_pred)
                
                dA = self.loss.backward()
                self.backward(dA)
                
                self.optimizer.step(self.layers)
                
                epoch_loss += loss * X_batch.shape[0]
                correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                total += X_batch.shape[0]
            
            avg_loss = epoch_loss / num_samples
            accuracy = correct / total * 100
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(accuracy)
            
            if validation_data is not None:
                val_loss, val_acc = self._validate(validation_data[0], validation_data[1])
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
            else:
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%')
        
        return self.history

    def _validate(self, X, y):
        """
        Validate the model on given data.
        
        Args:
            X: Validation data.
            y: Validation labels.
        
        Returns:
            Tuple of (loss, accuracy).
        """
        self._set_training_mode(False)
        y_pred = self.forward(X)
        loss = self.loss.forward(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, accuracy

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data of shape (num_samples, input_features).
        
        Returns:
            Predicted probabilities of shape (num_samples, num_classes).
        """
        self._set_training_mode(False)
        return self.forward(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on a test set.
        
        Args:
            X: Test data.
            y: True labels (one-hot encoded).
        
        Returns:
            Tuple of (loss, accuracy).
        """
        self._set_training_mode(False)
        y_pred = self.predict(X)
        loss = self.loss.forward(y, y_pred)
        
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels) * 100
        
        print(f'Evaluation Loss: {loss:.4f} - Accuracy: {accuracy:.2f}%')
        return loss, accuracy

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        print("=" * 60)
        print(f"{'Layer':<25} {'Output Shape':<20} {'Params':<15}")
        print("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'W'):
                shape = f"(None, {layer.W.shape[1]})"
                params = np.prod(layer.W.shape) + np.prod(layer.b.shape)
                total_params += params
                trainable_params += params
            elif hasattr(layer, 'gamma'):
                shape = f"(None, {layer.gamma.shape[1]})"
                params = np.prod(layer.gamma.shape) + np.prod(layer.beta.shape)
                total_params += params
                trainable_params += params
            else:
                shape = "-"
                params = 0
            
            print(f"{layer_name:<25} {shape:<20} {params:<15}")
        
        print("=" * 60)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print("=" * 60)

    def save(self, path):
        """
        Save the model weights to a file.
        
        Args:
            path: Path to the file where weights will be saved.
        """
        weights_bias = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights_bias[f'W{i + 1}'] = layer.W
                weights_bias[f'b{i + 1}'] = layer.b
            if hasattr(layer, 'gamma'):
                weights_bias[f'gamma{i + 1}'] = layer.gamma
                weights_bias[f'beta{i + 1}'] = layer.beta
                weights_bias[f'running_mean{i + 1}'] = layer.running_mean
                weights_bias[f'running_var{i + 1}'] = layer.running_var
        np.savez(path, **weights_bias)

    def load(self, path):
        """
        Load model weights from a file.
        
        Args:
            path: Path to the file containing saved weights.
        """
        data = np.load(path)
        weights_bias = {key: data[key] for key in data.files}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                key_W = f'W{i + 1}'
                key_b = f'b{i + 1}'
                if key_W in weights_bias:
                    layer.W = weights_bias[key_W]
                    layer.b = weights_bias[key_b]
            if hasattr(layer, 'gamma'):
                if f'gamma{i + 1}' in weights_bias:
                    layer.gamma = weights_bias[f'gamma{i + 1}']
                    layer.beta = weights_bias[f'beta{i + 1}']
                    layer.running_mean = weights_bias[f'running_mean{i + 1}']
                    layer.running_var = weights_bias[f'running_var{i + 1}']