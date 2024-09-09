import numpy as np

class Tanh:
    def forward(self, input):
        """
        Compute the forward pass of the Tanh activation function.

        Args:
            input (numpy.ndarray): The input array for the Tanh activation.

        Returns:
            numpy.ndarray: The output of the Tanh activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = input
        # Compute the Tanh activation: np.tanh(input)
        self.output = np.tanh(self.input)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Tanh activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the Tanh function
        # The gradient is: gradient_output * (1 - np.power(output, 2))
        self.diffv = gradient_output * (1.0 - np.power(self.output, 2))
        return self.diffv