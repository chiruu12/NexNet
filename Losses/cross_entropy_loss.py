import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss with built-in softmax for multi-class classification.
    
    Combines softmax activation and negative log-likelihood loss for
    numerical stability and efficient gradient computation.
    """
    
    def __init__(self, epsilon=1e-5):
        """
        Initialize the Cross-Entropy Loss.
        
        Args:
            epsilon: Small constant to prevent numerical instability from log(0).
        """
        self.epsilon = epsilon
        self.softmax = None
        self.targets = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Cross-Entropy Loss.
        
        Args:
            targets: True labels (one-hot encoded) of shape (batch_size, num_classes).
            predictions: Raw logits of shape (batch_size, num_classes).
        
        Returns:
            The computed cross-entropy loss (scalar).
        """
        p_max = np.max(predictions, axis=1, keepdims=True)
        exps = np.exp(predictions - p_max)
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        self.targets = targets

        batch_size = predictions.shape[0]
        self.loss = -np.sum(targets * np.log(self.softmax + self.epsilon)) / batch_size
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Cross-Entropy Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.targets.shape[0]
        return (self.softmax - self.targets) / batch_size