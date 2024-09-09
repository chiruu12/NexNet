import numpy as np

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