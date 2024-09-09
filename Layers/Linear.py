import numpy as np
from utils import Initializer

class Linear:
    def __init__(self, input_dim, output_dim, activation=None, initializer=None):
        """
        Initialize the Linear layer with random weights and optional activation function.

        Parameters:
        input_dim : Dimension of the input data.
        output_dim : Dimension of the output data.
        activation : Activation function object with `forward` and `backward` methods.
        initializer : Optional initializer object for weight initialization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initializer = Initializer(initializer) if initializer else Initializer()
        
        # Initialize weights and biases
        self.W = self.initializer.initialize_weights(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Perform the forward pass of the Linear layer followed by the activation function (if set).

        Parameters:
        X : Input data.

        Returns:
        Output of the forward pass.
        """
        self.input = X
        linear_output = np.dot(X, self.W) + self.b
        
        if self.activation:
            return self.activation.forward(linear_output)  #use foward function of activation function
        else:
            return linear_output  # No activation function

    def backward(self, dA):
        """
        Perform the backward pass of the Linear layer.

        Parameters:
        dA : Gradient of the loss with respect to the output.

        Returns:
        Gradient of the loss with respect to the input.
        """
        if self.activation:
            dZ = self.activation.backward(dA)  # sue Backward function of activation function
        else:
            dZ = dA  # No activation function 
        
        self.dW = np.dot(self.input.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)
