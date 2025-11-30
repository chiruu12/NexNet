import numpy as np


class CosineSimilarityLoss:
    """
    Cosine Similarity Loss for measuring angular distance between vectors.
    
    Loss is computed as 1 - cosine_similarity, where cosine similarity
    measures the cosine of the angle between two vectors.
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Initialize the Cosine Similarity Loss.
        
        Args:
            epsilon: Small constant to prevent division by zero.
        """
        self.epsilon = epsilon
        self.predictions = None
        self.targets = None
        self.dot = None
        self.norm_pred = None
        self.norm_tar = None
        self.simi = None

    def forward(self, predictions, targets):
        """
        Compute the forward pass of the Cosine Similarity Loss.
        
        Args:
            predictions: Predicted values of shape (batch_size, num_features).
            targets: True values of shape (batch_size, num_features).
        
        Returns:
            The computed Cosine Similarity Loss (scalar).
        """
        self.predictions = predictions
        self.targets = targets
        
        self.dot = np.sum(self.predictions * self.targets, axis=1)
        self.norm_pred = np.linalg.norm(self.predictions, axis=1)
        self.norm_tar = np.linalg.norm(self.targets, axis=1)
        
        self.simi = self.dot / (self.norm_pred * self.norm_tar + self.epsilon)
        self.loss = 1 - np.mean(self.simi)
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Cosine Similarity Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.predictions.shape[0]
        
        norm_pred_expanded = self.norm_pred[:, np.newaxis]
        norm_tar_expanded = self.norm_tar[:, np.newaxis]
        dot_expanded = self.dot[:, np.newaxis]
        
        grad_pred = (
            self.targets / (norm_pred_expanded * norm_tar_expanded + self.epsilon) -
            dot_expanded * self.predictions / (norm_pred_expanded ** 3 * norm_tar_expanded + self.epsilon)
        ) / batch_size
        
        return -grad_pred