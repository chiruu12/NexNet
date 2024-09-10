import numpy as np
class ELU:
    def __init__(self,alpha=1) -> None:
        """
        Initialize the ELU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        # Initialize alpha as a learnable parameter
        self.alpha = np.full_like(0.01, alpha) # it is used like this because it will be used for CNN's as well
    def forward(self, inputs):
        """
        Compute the forward pass of the ELU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the ELU activation.

        Returns:
            numpy.ndarray: The output of the ELU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the ELU activation: max(0, input)
        self.output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
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
        # If the input was positive, the gradient is 1; otherwise, it is constant * exp(input)
        # genrally more computationally expensive 
        self.diffv = np.where(self.input > 0, gradient_output, gradient_output * self.alpha * np.exp(self.input))
        return self.diffv