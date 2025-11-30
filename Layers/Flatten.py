import numpy as np


class Flatten:
    """
    Flatten layer that reshapes input to 2D.
    
    Useful for transitioning from convolutional layers to fully connected layers.
    """
    
    def __init__(self):
        """Initialize the Flatten layer."""
        self.input_shape = None

    def forward(self, inputs):
        """
        Flatten the input to 2D.
        
        Args:
            inputs: Input data of shape (batch_size, ...).
        
        Returns:
            Flattened output of shape (batch_size, num_features).
        """
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)

    def backward(self, gradient_output):
        """
        Reshape the gradient back to the original input shape.
        
        Args:
            gradient_output: Gradient of shape (batch_size, num_features).
        
        Returns:
            Gradient reshaped to original input shape.
        """
        return gradient_output.reshape(self.input_shape)
