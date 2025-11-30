from core.backend import get_array_module, zeros
from utils import Initializer


class Linear:
    """
    Fully connected (dense) layer that performs a linear transformation.
    
    Computes output = input @ W + b, optionally followed by an activation function.
    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """
    
    def __init__(self, input_dim, output_dim, activation=None, initializer=None):
        """
        Initialize the Linear layer.
        
        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            activation: Optional activation function with forward and backward methods.
            initializer: Weight initialization method ('xavier', 'he', 'random', 'zero').
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initializer = Initializer(initializer) if initializer else Initializer()
        
        self.W = self.initializer.initialize_weights(input_dim, output_dim)
        self.b = zeros((1, output_dim))
        
        self.input = None
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Perform the forward pass.
        
        Args:
            X: Input data of shape (batch_size, input_dim).
        
        Returns:
            Output of shape (batch_size, output_dim), optionally activated.
        """
        xp = get_array_module(X)
        self.input = X
        linear_output = xp.dot(X, self.W) + self.b
        
        if self.activation:
            return self.activation.forward(linear_output)
        return linear_output

    def backward(self, dA):
        """
        Perform the backward pass.
        
        Args:
            dA: Gradient of the loss with respect to the layer output.
        
        Returns:
            Gradient of the loss with respect to the layer input.
        """
        xp = get_array_module(dA)
        
        if self.activation:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA
        
        self.dW = xp.dot(self.input.T, dZ)
        self.db = xp.sum(dZ, axis=0, keepdims=True)
        return xp.dot(dZ, self.W.T)
