import numpy as np
class PReLU:
    def __init__(self, alpha=0.01):
        """
        Initialize the PReLU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        # Initialize alpha as a learnable parameter
        self.alpha = np.full_like(0.01, alpha) # it is used like this because it will be used for CNN's as well
        self.alpha_grad = 0  # To store the gradient of alpha

    def forward(self, inputs):
        """
        Compute the forward pass of the PReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the PReLU activation.

        Returns:
            numpy.ndarray: The output of the PReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the PReLU activation
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the PReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the PReLU function
        self.diffv = np.where(self.input > 0, gradient_output, self.alpha * gradient_output)
        
        # Compute the gradient with respect to alpha
        # For negative inputs,the gradient with respect to alpha is the sum of the gradient output
        self.alpha_grad = np.sum(np.where(self.input <= 0, gradient_output * self.input, 0))
        
        return self.diffv