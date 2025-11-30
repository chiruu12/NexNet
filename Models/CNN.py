import numpy as np


class CNN:
    """
    Convolutional Neural Network model for image classification and processing.
    
    This model provides a high-level interface for building and training CNNs
    with convolutional layers, pooling layers, and fully connected layers.
    
    Parameters
    ----------
    layers : list
        List of layer objects (Conv2D, MaxPool2D, Flatten, Linear, etc.)
        
    Attributes
    ----------
    layers : list
        List of layers in the model.
    training : bool
        Whether the model is in training mode.
    history : dict
        Training history with loss and accuracy values.
        
    Examples
    --------
    >>> from Models.CNN import CNN
    >>> from Layers.Conv2D import Conv2D
    >>> from Layers.Pooling import MaxPool2D
    >>> from Layers.Flatten import Flatten
    >>> from Layers.Linear import Linear
    >>> from Activation_classes.relu import ReLu
    >>> from Activation_classes.softmax import Softmax
    >>> 
    >>> model = CNN([
    ...     Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    ...     ReLu(),
    ...     MaxPool2D(pool_size=2, stride=2),
    ...     Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    ...     ReLu(),
    ...     MaxPool2D(pool_size=2, stride=2),
    ...     Flatten(),
    ...     Linear(64 * 7 * 7, 128),
    ...     ReLu(),
    ...     Linear(128, 10),
    ...     Softmax()
    ... ])
    """
    
    def __init__(self, layers=None):
        """
        Initialize the CNN model.
        
        Parameters
        ----------
        layers : list, optional
            List of layer objects. Default is None (empty model).
        """
        self.layers = layers if layers is not None else []
        self.training = True
        self.optimizer = None
        self.loss_fn = None
        self._compiled = False
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def add(self, layer):
        """
        Add a layer to the model.
        
        Parameters
        ----------
        layer : Layer object
            Layer to add.
            
        Returns
        -------
        self : CNN
            Returns self for method chaining.
        """
        self.layers.append(layer)
        return self
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : ndarray
            Input data of shape (batch_size, channels, height, width).
            
        Returns
        -------
        ndarray
            Output predictions.
        """
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = self.training
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        """
        Backward pass through the network.
        
        Parameters
        ----------
        grad : ndarray
            Gradient from the loss function.
            
        Returns
        -------
        ndarray
            Gradient with respect to input.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def __call__(self, x):
        """Make model callable."""
        return self.forward(x)
    
    def train(self):
        """Set model to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
    
    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
    
    def compile(self, optimizer, loss):
        """
        Configure the model for training.
        
        Parameters
        ----------
        optimizer : Optimizer
            Optimizer for updating weights.
        loss : Loss
            Loss function for computing gradients.
        """
        self.optimizer = optimizer
        self.loss_fn = loss
        self._compiled = True
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, verbose=True,
            callbacks=None, shuffle=True, clip_grad_norm=None, clip_grad_value=None):
        """
        Train the model.
        
        Parameters
        ----------
        X : ndarray
            Training images of shape (n_samples, channels, height, width).
        y : ndarray
            Target labels.
        epochs : int, optional
            Number of training epochs. Default is 10.
        batch_size : int, optional
            Mini-batch size. Default is 32.
        validation_data : tuple, optional
            Tuple of (X_val, y_val) for validation.
        verbose : bool, optional
            Print training progress. Default is True.
        callbacks : list, optional
            List of callback objects.
        shuffle : bool, optional
            Shuffle data each epoch. Default is True.
        clip_grad_norm : float, optional
            Maximum gradient norm for clipping.
        clip_grad_value : float, optional
            Maximum gradient value for clipping.
            
        Returns
        -------
        dict
            Training history.
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile(optimizer, loss)")
        
        from utils.grad_clip import clip_grad_norm as cgn, clip_grad_value as cgv
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            self.train()
            
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0.0
            epoch_correct = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                output = self.forward(X_batch)
                loss = self.loss_fn.forward(y_batch, output)
                grad = self.loss_fn.backward()
                self.backward(grad)
                
                if clip_grad_norm is not None:
                    cgn(self.layers, clip_grad_norm)
                if clip_grad_value is not None:
                    cgv(self.layers, clip_grad_value)
                
                self.optimizer.step(self.layers)
                
                epoch_loss += loss * (end_idx - start_idx)
                
                if y_batch.ndim == 1:
                    predictions = np.argmax(output, axis=1)
                    epoch_correct += np.sum(predictions == y_batch)
                else:
                    predictions = np.argmax(output, axis=1)
                    targets = np.argmax(y_batch, axis=1)
                    epoch_correct += np.sum(predictions == targets)
                
                if verbose and batch_idx % max(1, n_batches // 10) == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{n_batches}", end="")
            
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            val_str = ""
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1])
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                val_str = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            
            if verbose:
                print(f"\rEpoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}{val_str}")
            
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': train_loss, 'acc': train_acc})
        
        return self.history
    
    def evaluate(self, X, y, batch_size=32):
        """
        Evaluate the model.
        
        Parameters
        ----------
        X : ndarray
            Test images.
        y : ndarray
            Test labels.
        batch_size : int, optional
            Batch size for evaluation.
            
        Returns
        -------
        tuple
            (loss, accuracy)
        """
        self.eval()
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        total_correct = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            output = self.forward(X_batch)
            loss = self.loss_fn.forward(y_batch, output)
            
            total_loss += loss * (end_idx - start_idx)
            
            if y_batch.ndim == 1:
                predictions = np.argmax(output, axis=1)
                total_correct += np.sum(predictions == y_batch)
            else:
                predictions = np.argmax(output, axis=1)
                targets = np.argmax(y_batch, axis=1)
                total_correct += np.sum(predictions == targets)
        
        return total_loss / n_samples, total_correct / n_samples
    
    def predict(self, X, batch_size=32):
        """
        Generate predictions.
        
        Parameters
        ----------
        X : ndarray
            Input images.
        batch_size : int, optional
            Batch size for prediction.
            
        Returns
        -------
        ndarray
            Model predictions.
        """
        self.eval()
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            output = self.forward(X_batch)
            predictions.append(output)
        
        return np.vstack(predictions)
    
    def num_parameters(self):
        """
        Count total trainable parameters.
        
        Returns
        -------
        int
            Total number of parameters.
        """
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                total += layer.W.size
            if hasattr(layer, 'b') and layer.b is not None:
                total += layer.b.size
            if hasattr(layer, 'K') and layer.K is not None:
                total += layer.K.size
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                total += layer.gamma.size
            if hasattr(layer, 'beta') and layer.beta is not None:
                total += layer.beta.size
        return total
    
    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print(f"{'Layer':<30} {'Output Shape':<20} {'Params':<15}")
        print("=" * 70)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}"
            
            params = 0
            if hasattr(layer, 'W') and layer.W is not None:
                params += layer.W.size
            if hasattr(layer, 'b') and layer.b is not None:
                params += layer.b.size
            if hasattr(layer, 'K') and layer.K is not None:
                params += layer.K.size
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                params += layer.gamma.size
            if hasattr(layer, 'beta') and layer.beta is not None:
                params += layer.beta.size
            
            shape_str = "?"
            if hasattr(layer, 'out_channels'):
                shape_str = f"(None, {layer.out_channels}, ?, ?)"
            elif hasattr(layer, 'output_size'):
                shape_str = f"(None, {layer.output_size})"
            elif hasattr(layer, 'pool_size'):
                shape_str = "(None, ?, ?, ?)"
            
            print(f"{layer_name:<30} {shape_str:<20} {params:<15}")
            total_params += params
        
        print("=" * 70)
        print(f"Total parameters: {total_params:,}")
        print("=" * 70)
    
    def save(self, filepath):
        """
        Save model weights.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        state = {}
        for i, layer in enumerate(self.layers):
            layer_state = {}
            if hasattr(layer, 'W') and layer.W is not None:
                layer_state['W'] = layer.W
            if hasattr(layer, 'b') and layer.b is not None:
                layer_state['b'] = layer.b
            if hasattr(layer, 'K') and layer.K is not None:
                layer_state['K'] = layer.K
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                layer_state['gamma'] = layer.gamma
            if hasattr(layer, 'beta') and layer.beta is not None:
                layer_state['beta'] = layer.beta
            if hasattr(layer, 'running_mean') and layer.running_mean is not None:
                layer_state['running_mean'] = layer.running_mean
            if hasattr(layer, 'running_var') and layer.running_var is not None:
                layer_state['running_var'] = layer.running_var
            if layer_state:
                state[f'layer_{i}'] = layer_state
        
        np.savez(filepath, **{k: v for d in state.values() for k, v in d.items()})
    
    def load(self, filepath):
        """
        Load model weights.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from.
        """
        data = np.load(filepath, allow_pickle=True)
        
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W') and f'arr_{param_idx}' in data.files:
                layer.W = data[f'arr_{param_idx}']
                param_idx += 1
            if hasattr(layer, 'b') and f'arr_{param_idx}' in data.files:
                layer.b = data[f'arr_{param_idx}']
                param_idx += 1
            if hasattr(layer, 'K') and f'arr_{param_idx}' in data.files:
                layer.K = data[f'arr_{param_idx}']
                param_idx += 1
    
    def __repr__(self):
        """String representation."""
        layer_strs = [f"  ({i}): {layer.__class__.__name__}" for i, layer in enumerate(self.layers)]
        return f"CNN(\n" + "\n".join(layer_strs) + "\n)"
