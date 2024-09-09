import numpy as np
class LeakyReLu:
    def __int__(self,alpha=0.01):
        """
        Initialize the LeakyReLU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        self.alpha=alpha
    def forward(self, inputs):
        """
        Compute the forward pass of the LeakyReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the LeakyReLU activation.

        Returns:
            numpy.ndarray: The output of the LeakyReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the LeakyReLU activation: max(small_constant*input, input)
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
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
        # If the input was positive, the gradient is 1; otherwise, it is the aplha contstant 
        self.diffv = np.where(self.input > 0, gradient_output, self.alpha * gradient_output)
        return self.diffv