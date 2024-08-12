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


class Softmax:
    def forward(self, inputs):
        """
        Perform the forward pass of the Softmax activation function.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Output data after applying the Softmax activation function, of the same shape as the input.
        """
        # Compute exponentiated values, with numerical stability
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize by the sum of all the exp values
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, d_outputs):
        """
        Perform the backward pass of the Softmax activation function to compute gradients.

        Args:
            d_outputs (np.ndarray): Gradient of the loss with respect to the output of this layer, of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer, of the same shape as the input.
        """
        batch_size = self.outputs.shape[0]
        num_classes = self.outputs.shape[1]

        # Initialize gradient of the input
        d_inputs = np.zeros_like(d_outputs)

        # Compute gradients for each sample in the batch
        for i in range(batch_size):
            single_output = self.outputs[i].reshape(-1, 1)
            single_grad_output = d_outputs[i]

            # Jacobian matrix for the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Compute gradient for the current sample
            d_inputs[i] = np.dot(jacobian_matrix, single_grad_output)

        return d_inputs
