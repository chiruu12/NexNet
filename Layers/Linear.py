import numpy as np
from utils import Initializer
class Linear:
    def __init__(self, input_dim, output_dim,initializer=None):
        """
        Initialize the Linear layer with random weights and zero biases.

        Parameters:
        input_dim : Dimension of the input data.
        output_dim : Dimension of the output data.
        """
        self.initializer=Initializer(initializer) if initializer!=None else Initializer()
        self.W = self.initializer.initialize_weights(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Perform the forward pass of the Linear layer.

        Parameters:
        X : Input data.

        Returns:
        Output of the forward pass.
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, dA):
        """
        Perform the backward pass of the Linear layer.

        Parameters:
        dA : the loss with respect to the output.

        Returns:
        Gradient of the loss with respect to the input.
        """
        self.dW = np.dot(self.input.T, dA)
        self.db = np.sum(dA, axis=0, keepdims=True)
        return np.dot(dA, self.W.T)