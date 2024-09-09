import numpy as np
class ReLu:
    def forward(self, inputs):
        """
        Compute the forward pass of the ReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the ReLU activation.

        Returns:
            numpy.ndarray: The output of the ReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the ReLU activation: max(0, input)
        self.output = np.maximum(0, inputs)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the ReLU function
        # If the input was positive, the gradient is 1; otherwise, it is 0
        self.diffv = np.where(self.input > 0, gradient_output, 0)
        return self.diffv
