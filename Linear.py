import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        """
        Initialize the Linear layer with random weights and zero biases.

        Parameters:
        input_dim (int): Dimension of the input data.
        output_dim (int): Dimension of the output data.
        """
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Perform the forward pass of the Linear layer.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output of the forward pass.
        """
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dA):
        """
        Perform the backward pass of the Linear layer.

        Parameters:
        dA (numpy.ndarray): Gradient of the loss with respect to the output.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input.
        """
        m = self.X.shape[0]
        self.dW = np.dot(self.X.T, dA) / m
        self.db = np.sum(dA, axis=0, keepdims=True) / m
        return np.dot(dA, self.W.T)