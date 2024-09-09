import numpy as np
class CosineSimilarityLoss:
    def __init__(self, epsilon=1e-8):
        """
        Initialize the Cosine Similarity Loss class.

        Args:
            epsilon : to prevent division by zero and edge cases.
        """
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Compute the forward pass of the Cosine Similarity Loss function.

        Args:
            targets : True values, of shape (batch_size, num_features).
            predictions : Predicted values, of shape (batch_size, num_features).

        Returns:
            The computed Cosine Similarity Loss.
        """
        self.pred = predictions
        self.tar = targets
        # Compute the dot product
        self.dot = np.sum(self.pred *self.tar, axis=1)
        # Compute norms
        self.norm_pred = np.linalg.norm(self.predictions, axis=1)
        self.norm_tar = np.linalg.norm(self.targets, axis=1)

        # Compute cosine similarity and then the loss which is = 1 - similarity
        self.simi= self.dot/(self.norm_pred * self.norm_tar + self.epsilon)
        self.loss = 1 - np.mean(self.simi)
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Cosine Similarity Loss function.

        Returns:
            Gradient of the loss with respect to the predictions, and targets.
        """
        batch_size = self.predictions.shape[0]
        # calculate gradient
        grad_pred = (self.targets / (self.norm_tar + self.epsilon) - 
        (self.dot/ (self.norm_tar**2 + self.epsilon))*(self.predictions /(self.norm_pred + self.epsilon)))/batch_size

        return grad_pred