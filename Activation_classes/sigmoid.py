import numpy as np

class Sigmoid:
    def forward(self, input):
        """
        Compute the forward pass of the Sigmoid activation function.

        Args:
            input (numpy.ndarray): The input array for the Sigmoid activation.

        Returns:
            numpy.ndarray: The output of the Sigmoid activation (same shape as input).
        """
        # Compute the Sigmoid activation: 1 / (1 + exp(-input))
        self.output = 1 / (1 + np.exp(-input))
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Sigmoid activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the Sigmoid function
        # The gradient is: gradient_output * (1 - output) * output
        self.diffv = gradient_output * (1 - self.output) * self.output
        return self.diffv
