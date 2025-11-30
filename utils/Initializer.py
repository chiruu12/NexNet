from core.backend import get_array_module, zeros, random_randn


class Initializer:
    """
    Weight initializer for neural network layers.
    
    Provides various initialization strategies to help with training
    stability and convergence. Supports both CPU and GPU backends.
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
        xp = get_array_module()
        shape = (input_dim, output_dim)
        
        if self.method == 'xavier':
            limit = xp.sqrt(6 / (input_dim + output_dim))
            return xp.random.uniform(-limit, limit, size=shape)

        elif self.method == 'he':
            stddev = xp.sqrt(2. / input_dim)
            return random_randn(*shape) * stddev

        elif self.method == 'random':
            return xp.random.uniform(-0.1, 0.1, size=shape)

        elif self.method == 'zero':
            return zeros(shape)

        else:
            raise ValueError(f"Unsupported initialization method: {self.method}")
