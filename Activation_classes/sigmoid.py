from core.backend import get_array_module, zeros_like, exp


class Sigmoid:
    """
    Sigmoid activation function.
    
    Maps input values to the range (0, 1), useful for binary
    classification and gates in recurrent networks.
    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """
    
    def forward(self, inputs):
        """
        Compute the forward pass of the Sigmoid activation function.

        Args:
            inputs: The input array for the Sigmoid activation.

        Returns:
            The output of the Sigmoid activation (same shape as input).
        """
        self.inputs = inputs
        self.output = self._stable_sigmoid(inputs)
        return self.output

    def _stable_sigmoid(self, x):
        """
        Numerically stable sigmoid implementation.
        
        Handles both large positive and large negative values
        to prevent overflow in exp().
        """
        xp = get_array_module(x)
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        
        result = zeros_like(x)
        
        result[positive_mask] = 1 / (1 + exp(-x[positive_mask]))
        
        exp_x = exp(x[negative_mask])
        result[negative_mask] = exp_x / (1 + exp_x)
        
        return result

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Sigmoid activation function.

        Args:
            gradient_output: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        self.diffv = gradient_output * self.output * (1 - self.output)
        return self.diffv
