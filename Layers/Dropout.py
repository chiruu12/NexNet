from core.backend import get_array_module, random_rand


class Dropout:
    """
    Dropout layer for regularization during training.
    
    Randomly sets a fraction of input units to zero during training,
    which helps prevent overfitting. During inference, all units are used
    but scaled appropriately.
    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """
    
    def __init__(self, rate=0.5):
        """
        Initialize the Dropout layer.
        
        Args:
            rate: Fraction of input units to drop (between 0 and 1).
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in range [0, 1)")
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, inputs):
        """
        Perform the forward pass of the Dropout layer.
        
        Args:
            inputs: Input data of shape (batch_size, features).
        
        Returns:
            Output with dropout applied during training, or scaled input during inference.
        """
        if self.training:
            xp = get_array_module(inputs)
            self.mask = (random_rand(*inputs.shape) > self.rate).astype(inputs.dtype) / (1 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, gradient_output):
        """
        Perform the backward pass of the Dropout layer.
        
        Args:
            gradient_output: Gradient of the loss with respect to the output.
        
        Returns:
            Gradient of the loss with respect to the input.
        """
        if self.training:
            return gradient_output * self.mask
        return gradient_output

    def train(self):
        """Set the layer to training mode."""
        self.training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False
