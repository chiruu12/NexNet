import numpy as np
from .sigmoid import Sigmoid
class Swish:
    def __init__(self):
        """
        Initialize the Swish activation function.
        """
        self.sigmoid = Sigmoid()
        

    def forward(self, inputs):
        """
        Compute the forward pass of the Swish activation function.

        Args:
            inputs (numpy.ndarray): The input array for the Swish activation.

        Returns:
            numpy.ndarray: The output of the Swish activation (same shape as input).
        """
        self.inputs=inputs
        # Compute the sigmoid of the inputs using the Sigmoid class
        self.sigmoid_output = self.sigmoid.forward(inputs)
        # Compute the Swish activation
        self.output = inputs*self.sigmoid_output
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Swish activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Get the gradient of the sigmoid output
        grad_sigmoid_output = self.sigmoid.backward(gradient_output)
        # Compute the gradient of the Swish function
        grad_input = gradient_output*(self.sigmoid_output + self.inputs*grad_sigmoid_output)
        return grad_input
    
