import numpy as np


class MAE:
    """
    Mean Absolute Error (L1 Loss) for regression tasks.
    
    Calculates the average of absolute differences between predictions and targets.
    More robust to outliers than MSE.
    """
    
    def __init__(self):
        """Initialize the MAE Loss."""
        self.predictions = None
        self.targets = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Mean Absolute Error Loss.
        
        Args:
            targets: True values of shape (batch_size,) or (batch_size, features).
            predictions: Predicted values of same shape as targets.
        
        Returns:
            The computed MAE loss (scalar).
        """
        self.predictions = predictions
        self.targets = targets
        self.loss = np.mean(np.abs(predictions - targets))
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Mean Absolute Error Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.targets.size
        grad = np.sign(self.predictions - self.targets) / batch_size
        return grad
