from core.backend import get_array_module, log, maximum, mean


class PoissonLoss:
    """
    Poisson Loss for count-based prediction tasks.
    
    Measures the difference between predicted rates and actual event counts
    using the negative log-likelihood of the Poisson distribution.
    """
    
    def __init__(self):
        """Initialize the Poisson Loss."""
        self.predictions = None
        self.targets = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Poisson Loss.
        
        Args:
            targets: True event counts of shape (batch_size,).
            predictions: Predicted rates (lambda) of shape (batch_size,).
        
        Returns:
            The computed Poisson loss (scalar).
        """
        self.predictions = maximum(predictions, 1e-8)
        self.targets = targets
        
        self.loss = mean(self.predictions - targets * log(self.predictions))
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Poisson Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        batch_size = self.targets.shape[0]
        grad = (1 - self.targets / self.predictions) / batch_size
        return grad