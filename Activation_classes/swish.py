from .sigmoid import Sigmoid


class Swish:
    """
    Swish activation function: x * sigmoid(x).
    
    A smooth, non-monotonic activation function that often outperforms ReLU.
    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """
    
    def __init__(self):
        """Initialize the Swish activation function."""
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        """
        Compute the forward pass of the Swish activation function.

        Args:
            inputs: The input array for the Swish activation.

        Returns:
            The output of the Swish activation (same shape as input).
        """
        self.inputs = inputs
        self.sigmoid_output = self.sigmoid.forward(inputs)
        self.output = inputs * self.sigmoid_output
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Swish activation function.

        Args:
            gradient_output: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        grad_sigmoid_output = self.sigmoid.backward(gradient_output)
        grad_input = gradient_output * (self.sigmoid_output + self.inputs * grad_sigmoid_output)
        return grad_input
