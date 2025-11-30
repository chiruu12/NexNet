from core.backend import get_array_module, maximum, where


class ReLu:
    """ReLU activation function: max(0, x). Supports CPU/GPU backends."""
    
    def forward(self, inputs):
        """
        Compute the forward pass of the ReLU activation function.

        Args:
            inputs: The input array for the ReLU activation.

        Returns:
            The output of the ReLU activation (same shape as input).
        """
        self.input = inputs
        self.output = maximum(inputs, 0)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ReLU activation function.

        Args:
            gradient_output: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        self.diffv = where(self.input > 0, gradient_output, 0)
        return self.diffv
