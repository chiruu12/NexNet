from core.backend import get_array_module, mean


class MSE:
    """
    Mean Squared Error (L2 Loss) for regression tasks.
    
    Calculates the average of squared differences between predictions and targets.
    Penalizes larger errors more heavily than MAE.
    """
    
    def __init__(self):
        """Initialize the MSE Loss."""
        self.predictions = None
        self.targets = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Mean Squared Error Loss.
        
        Args:
            targets: True values of shape (batch_size,) or (batch_size, features).
            predictions: Predicted values of same shape as targets.
        
        Returns:
            The computed MSE loss (scalar).
        """
        self.predictions = predictions
        self.targets = targets
        self.loss = mean((predictions - targets) ** 2)
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Mean Squared Error Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.targets.size
        grad = 2 * (self.predictions - self.targets) / batch_size
        return grad