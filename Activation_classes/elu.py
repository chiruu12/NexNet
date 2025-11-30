from core.backend import get_array_module, exp, where


class ELU:
    def __init__(self, alpha=1) -> None:
        """
        Initialize the ELU activation function with a learnable alpha parameter.

        Args:
            alpha (float): Value for the alpha parameter.
        """
        # Store alpha as a scalar - will be broadcast automatically
        self.alpha = alpha
        
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
        # Compute the ELU activation
        self.output = where(inputs > 0, inputs, self.alpha * (exp(inputs) - 1))
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ELU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the ELU function
        # If the input was positive, the gradient is 1; otherwise, it is alpha * exp(input)
        self.diffv = where(self.input > 0, gradient_output, gradient_output * self.alpha * exp(self.input))
        return self.diffv