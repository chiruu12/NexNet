import numpy as np
from .sigmoid import Sigmoid
class Softplus:
    def __init__(self):
        """
        Initialize the Softplus activation function.
        """
        self.sigmoid = Sigmoid()
        
    def forward(self, inputs):
        """
        Compute the forward pass of the Softplus activation function.

        Args:
            inputs (numpy.ndarray): The input array for the Softplus activation.

        Returns:
            numpy.ndarray: The output of the Softplus activation (same shape as input).
        """
        # Compute the Softplus activation
        self.input = inputs
        self.output = np.log(1 + np.exp(inputs))
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Softplus activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the sigmoid of the input using the Sigmoid class
        sigmoid_input = self.sigmoid.forward(self.input)
        # The gradient of Softplus is the sigmoid of the input
        grad_input = gradient_output * sigmoid_input
        return grad_input