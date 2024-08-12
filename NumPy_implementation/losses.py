import numpy as np

class CrossEntropyLoss:
    def forward(self, targets, predictions):
        """
        Perform the forward pass of the Cross-Entropy Loss function.

        Args:
            targets (np.ndarray): True labels, one-hot encoded, of shape (batch_size, num_classes).
            predictions (np.ndarray): Predicted probabilities, of shape (batch_size, num_classes).

        Returns:
            float: The computed cross-entropy loss.
        """
        # Ensure numerical stability by subtracting the max value from predictions
        p_max = np.max(predictions, axis=1, keepdims=True)
        exps = np.exp(predictions - p_max)
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        self.targets = targets

        # Compute the loss using the cross-entropy formula
        batch_size = predictions.shape[0]
        self.loss = -np.sum(targets * np.log(self.softmax + 1e-10)) / batch_size  # Add epsilon for numerical stability
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Cross-Entropy Loss function to compute gradients.

        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions, of shape (batch_size, num_classes).
        """
        batch_size = self.targets.shape[0]
        # Compute gradient of the loss with respect to the predictions
        d_predictions = (self.softmax - self.targets) / batch_size
        return d_predictions


    
class MeanSquaredErrorLoss:
    def forward(self, predictions, targets):
        """
        Perform the forward pass of the Mean Squared error Loss function.

        Args:
            targets (np.ndarray): True labels, one-hot encoded, of shape (batch_size, num_classes).
            predictions (np.ndarray): Predicted probabilities, of shape (batch_size, num_classes).

        Returns:
            float: The computed cross-entropy loss.
        """
        self.predictions = predictions
        self.targets = targets
        self.loss = np.mean((predictions - targets) ** 2)
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Mean Squared error Loss function.

        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions, of shape (batch_size, num_classes).
        """
        grad_input = 2 * (self.predictions - self.targets) / self.targets.size
        return grad_input