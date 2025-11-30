import numpy as np


class Initializer:
    """
    Weight initializer for neural network layers.
    
    Provides various initialization strategies to help with training
    stability and convergence.
    """
    
    def __init__(self, method='xavier'):
        """
        Initialize the Initializer.
        
        Args:
            method: Initialization method ('xavier', 'he', 'random', 'zero').
        """
        self.method = method

    def initialize_weights(self, input_dim, output_dim):
        """
        Initialize weights based on the specified method.
        
        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
        
        Returns:
            Initialized weight matrix of shape (input_dim, output_dim).
        """
        shape = (input_dim, output_dim)
        
        if self.method == 'xavier':
            limit = np.sqrt(6 / (input_dim + output_dim))
            return np.random.uniform(-limit, limit, size=shape)

        elif self.method == 'he':
            stddev = np.sqrt(2. / input_dim)
            return np.random.randn(*shape) * stddev

        elif self.method == 'random':
            return np.random.uniform(-0.1, 0.1, size=shape)

        elif self.method == 'zero':
            return np.zeros(shape)

        else:
            raise ValueError(f"Unsupported initialization method: {self.method}")
