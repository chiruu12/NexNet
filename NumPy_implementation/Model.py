class Model:
    def __init__(self):
        """
        Initialize the Model class.
        This class manages the layers, loss function, and optimizer for training and inference.
        """
        self.layers = []

    def add_layer(self, layer):
        """
        Add a layer to the model.

        Args:
            layer (object): A layer object that has `forward` and `backward` methods. The layer should 
                            also have attributes like `W` and `b` if it contains learnable parameters.
        """
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Compile the model by specifying the loss function and optimizer.

        Args:
            loss (object): An instance of a loss class that has `forward` and `backward` methods.
            optimizer (object): An instance of an optimizer class that has a `step` method.
        """
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        """
        Perform a forward pass through the model.

        Args:
            X (np.ndarray): Input data of shape (batch_size, ...).

        Returns:
            np.ndarray: The output of the model after passing through all layers.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        """
        Perform a backward pass through the model to compute gradients.

        Args:
            dA (np.ndarray): Gradient of the loss with respect to the model's output.

        Returns:
            None: The method updates the gradients of the layers in place.
        """
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def train(self, X, y, epochs, batch_size):
        """
        Train the model using mini-batch gradient descent.

        Args:
            X (np.ndarray): Training data of shape (num_samples, ...).
            y (np.ndarray): True labels, one-hot encoded, of shape (num_samples, num_classes).
            epochs (int): Number of training epochs.
            batch_size (int): Size of each mini-batch.
        """
        num_samples = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.loss.forward(y_batch, y_pred)

                # Backward pass
                dA = self.loss.backward()
                self.backward(dA)

                # Update weights and biases
                self.optimizer.step(self.layers)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Input data of shape (num_samples, ...).

        Returns:
            np.ndarray: Predicted probabilities of shape (num_samples, num_classes).
        """
        return self.forward(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on a test set.

        Args:
            X (np.ndarray): Test data of shape (num_samples, ...).
            y (np.ndarray): True labels, one-hot encoded, of shape (num_samples, num_classes).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Predicted probabilities of shape (num_samples, num_classes).
                - float: Loss value on the test set.
                - float: Accuracy percentage on the test set.
        """
        y_pred = self.predict(X)
        loss = self.loss.forward(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, accuracy

    def save(self, path):
        """
        Save the model weights to a file.

        Args:
            path (str): Path to the file where the model weights and bias will be saved.
        """
        weights_bias = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights_bias[f'W{i+1}'] = layer.W
                weights_bias[f'b{i+1}'] = layer.b
        np.savez(path, **weights_bias)

    def load(self, path):
        """
        Load a model from a file.

        Args:
            path (str): Path to the file where the model is saved.

        Returns:
            Model: The loaded model
        """
        data = np.load(path)
        weights_bias = {key: data[key] for key in data.files}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                key_W = f'W{i+1}'
                key_b = f'b{i+1}'
                layer.W = weights_bias[key_W]
                layer.b = weights_bias[key_b]