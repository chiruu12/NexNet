import numpy as np

class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss for binary classification tasks.
    
    Measures the performance of a model whose output is a probability between 0 and 1.
    """
    
    def __init__(self, epsilon=1e-5):
        """
        Initialize the Binary Cross-Entropy Loss.
        
        Args:
            epsilon: Small constant to prevent numerical instability from log(0).
        """
        self.epsilon = epsilon
        self.predictions = None
        self.targets = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Binary Cross-Entropy Loss.
        
        Args:
            targets: True binary labels of shape (batch_size,) or (batch_size, 1).
            predictions: Predicted probabilities of shape (batch_size,) or (batch_size, 1).
        
        Returns:
            The computed binary cross-entropy loss (scalar).
        """
        self.predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        self.targets = targets
        
        batch_size = targets.shape[0]
        self.loss = -np.sum(
            targets * np.log(self.predictions) + 
            (1 - targets) * np.log(1 - self.predictions)
        ) / batch_size
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Binary Cross-Entropy Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.targets.shape[0]
        diff_predictions = (self.predictions - self.targets) / (
            self.predictions * (1 - self.predictions) * batch_size
        )
        return diff_predictions